"""
bash — run an arbitrary shell command on the client computer.

Permission is required for every call unless the user has granted a
session-scoped exact or pattern-based allow for the specific command.
The tool is disabled by default because it gives the model unrestricted
access to the client machine.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shlex
import subprocess
from typing import TYPE_CHECKING, Any

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT_CHARS = 1024
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Matches a redirection operator token: bare (">", ">>", "<") or with an
# optional fd prefix ("2>", "2>>").
_REDIR_OP_RE = re.compile(r"^[0-9]*>>?$|^<$")
# Bare operator without an fd prefix — used to decide whether to absorb the
# preceding token as an fd number.
_REDIR_BARE_OP_RE = re.compile(r"^>>?$|^<$")
# A bare fd number token (e.g. "2" before a ">").
_REDIR_FD_RE = re.compile(r"^\d+$")
# An fd-redirect target in one token (e.g. "&1" in "2>&1").
_REDIR_FD_TARGET_RE = re.compile(r"^&\d+$")
# Shell metacharacters that, if present in a redirection target, could trigger
# command execution via shell expansion when the command runs with shell=True.
# Wildcard grants must never auto-approve such targets.
_REDIR_SHELL_META_RE = re.compile(r"[$`(){}!]")

_CAPTURE_MODES = ("stdout", "stderr", "interleaved", "separate")
# POSIX: inside double-quotes a backslash only escapes these characters.
_DQ_ESCAPE = frozenset('\\\n$`"')
# Characters that cannot begin a heredoc WORD delimiter.  Any character not
# in this set is accepted as a valid delimiter start, matching the shell
# grammar where a heredoc delimiter is any WORD token.
_HEREDOC_NON_WORD = frozenset(" \t\n|&;()<>`")
# Shell keywords that introduce a command position: after one of these words,
# the next token can be (( or [[ as a compound command.
_CMD_INTRODUCING_WORDS = frozenset(
    {
        "if",
        "elif",
        "while",
        "until",
        "for",
        "then",
        "do",
        "else",
        "!",
        "case",
        "select",
        "function",
        "time",
    }
)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Return *(text, truncated)* where *text* is at most *max_chars* characters."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _has_heredoc(segment: str) -> bool:
    """Return True if *segment* contains a shell heredoc operator (``<<``).

    Tracks quoting state (single-quoted, double-quoted, and ANSI-C
    ``$'...'``) and nesting depth for the specific bash compound constructs
    ``$((…))``, ``((…))``, and ``[[…]]`` so that ``<<`` used as an
    arithmetic or comparison operator is not mistaken for a heredoc.

    ``$((`` is tracked regardless of position (arithmetic expansion can
    appear mid-expression, e.g. ``echo $((1<<2))``).  ``((`` and ``[[``
    are tracked only when they appear at a *command-start* position —
    immediately after a chain operator, open parenthesis, or newline, or
    at the beginning of the segment.  After an ordinary word such as
    ``echo``, the same ``((`` / ``[[`` tokens are literal arguments and
    must not block subsequent heredoc detection.

    ANSI-C quoting (``$'...'``) is handled separately from plain
    ``'...'``: inside ``$'...'``, backslash escapes are honoured so that
    ``\\'`` does not close the string prematurely.

    When an unquoted ``<<`` is found outside those tracked constructs, the
    delimiter check accepts any character that is not whitespace or a shell
    metacharacter (``_HEREDOC_NON_WORD``).  This covers the full range of
    bash/POSIX heredoc delimiter forms: quoted strings (``'…'`` / ``"…"``),
    backslash-escaped characters (``\\EOF``), identifiers, digit-start words
    (``<<1``), dash-start words (``<< -EOF``), and parameter-expanded words
    (``<<$DELIM``).

    Additional tokens or operators after the delimiter (e.g. ``<<EOF | wc``,
    ``<<EOF >out.txt``) are ignored; the function returns True as soon as the
    delimiter is confirmed.
    """
    n = len(segment)
    i = 0
    in_single = False
    in_ansi_c = False  # True while inside $'...' ANSI-C quoting
    in_double = False
    depth = 0  # nesting depth of (( )) / [[ ]] / $(( )) constructs
    # True at the start of the segment and after chain operators, subshell
    # opens, or shell keywords that introduce a command position (if, then,
    # do, while, …).  (( and [[ only enter depth-tracking mode at command-
    # start positions; after an ordinary word they are literal arguments.
    at_cmd_start = True
    # Start position of the current unquoted word, or -1 when not in a word.
    # Used to detect reserved words at word boundaries.
    word_start: int = -1
    # True if the current word started when at_cmd_start was already True.
    # Only words that begin at a command position are treated as reserved words;
    # identical tokens used as arguments (e.g. "echo if [[") are not.
    word_started_at_cmd_start: bool = True
    # Set when the next word is a redirection target (e.g. after ">" or "<").
    # After that target word is consumed, at_cmd_start is restored to True so
    # that (( / [[ following a leading redirect are tracked correctly.
    skip_redir_target: bool = False

    def _end_word(pos: int) -> None:
        """Check if the word ending at *pos* is a command-introducing keyword."""
        nonlocal at_cmd_start, word_start, word_started_at_cmd_start, skip_redir_target
        if word_start >= 0:
            word = segment[word_start:pos]
            if skip_redir_target:
                # This word was a redirection target; we're back at cmd position.
                at_cmd_start = True
                skip_redir_target = False
            elif word_started_at_cmd_start:
                if word in _CMD_INTRODUCING_WORDS:
                    at_cmd_start = True
                elif _ENV_VAR_RE.match(word):
                    # Env-var assignment prefix: still at command position.
                    at_cmd_start = True
                elif _REDIR_OP_RE.match(word):
                    # Redirect operator (e.g. ">", ">>", "2>"): next word is
                    # the target; restore at_cmd_start after consuming it.
                    skip_redir_target = True
        word_start = -1

    while i < n:
        c = segment[i]
        if c == "\\" and (not in_single or in_ansi_c):
            # Outside quotes: skip any backslash-escaped character.
            # Inside $'...': backslash is also an escape (e.g. \' does not
            # close the string), so consume both chars to stay in-quote.
            i += 2
            continue
        if (
            c == "$"
            and not in_single
            and not in_double
            and i + 1 < n
            and segment[i + 1] == "'"
        ):
            # ANSI-C quoting: $'...' — backslash escapes are active inside.
            _end_word(i)
            in_single = True
            in_ansi_c = True
            i += 2
            continue
        if c == "'" and not in_double:
            _end_word(i)
            if in_single:
                in_single = False
                in_ansi_c = False
            else:
                in_single = True
                # in_ansi_c stays False: plain '...' does not honour escapes.
            # Entering or exiting a quote is part of a word token, so clear
            # command-start — a following [[ / (( is a literal argument.
            at_cmd_start = False
        elif c == '"' and not in_single:
            _end_word(i)
            in_double = not in_double
            at_cmd_start = False  # same: quote is part of a word token
        elif not in_single and not in_double:
            if c in " \t":
                # Horizontal whitespace ends the current word.  Check for a
                # command-introducing keyword; preserve at_cmd_start otherwise.
                _end_word(i)
                i += 1
                continue
            if c == "\n":
                _end_word(i)
                at_cmd_start = True
                i += 1
                continue
            # $((: arithmetic expansion — suppress heredoc detection regardless
            # of position (may appear mid-expression, e.g. echo $((1<<2))).
            if c == "$" and segment.startswith("$((", i):
                _end_word(i)
                depth += 1
                at_cmd_start = False
                i += 3
                continue
            # )) or ]]: close the innermost tracked construct.
            if depth > 0 and (
                segment.startswith("))", i) or segment.startswith("]]", i)
            ):
                word_start = -1
                depth = max(0, depth - 1)
                i += 2
                continue
            # ( and [: may be the start of (( or [[.  Call _end_word() BEFORE
            # consulting at_cmd_start so that "if((" (no space between keyword
            # and compound command) is handled correctly — _end_word("if")
            # sets at_cmd_start=True, then the (( check fires at the same
            # character position.
            if c in "([":
                _end_word(i)
                if at_cmd_start and segment.startswith("((", i):
                    depth += 1
                    at_cmd_start = False
                    i += 2
                    continue
                if at_cmd_start and segment.startswith("[[", i):
                    depth += 1
                    at_cmd_start = False
                    i += 2
                    continue
                # Lone ( or [ — subshell open or standalone bracket.
                # Inside a tracked construct (depth > 0) they may be nested
                # parentheses in arithmetic (e.g. $(( (1+2) ))) and must not
                # re-enable (( or [[ openers at the next character.
                if depth == 0:
                    at_cmd_start = True
            # Other chain operators and open-brace reset command-start.
            elif c in "|&;{":
                _end_word(i)
                at_cmd_start = True
            elif c == "<" and depth == 0:
                _end_word(i)
                # Count consecutive unquoted '<' characters.
                j = i
                while j < n and segment[j] == "<":
                    j += 1
                if j - i == 2:  # exactly "<<" — candidate heredoc operator
                    k = j
                    # Optional heredoc-strip flag.
                    if k < n and segment[k] == "-":
                        k += 1
                    # Skip optional horizontal whitespace before the delimiter.
                    while k < n and segment[k] in " \t":
                        k += 1
                    # Accept any character that can begin a shell WORD — i.e.
                    # anything that is not whitespace or a shell metacharacter.
                    if k < n and segment[k] not in _HEREDOC_NON_WORD:
                        return True
                elif j - i == 1:  # single "<" — stdin redirect; target follows
                    # Don't clear at_cmd_start: after the target word, we restore
                    # it to True so that (( / [[ after a leading redirect are
                    # correctly identified as compound commands.
                    skip_redir_target = True
                    i = j
                    continue
                at_cmd_start = False
                i = j
                continue
            else:
                # Regular word character.
                if word_start < 0:
                    word_started_at_cmd_start = at_cmd_start
                    word_start = i
                at_cmd_start = False
        i += 1
    return False


if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace


def _split_env_vars(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    """Split leading KEY=val tokens from *tokens*, returning (env_dict, remainder)."""
    env: dict[str, str] = {}
    for i, token in enumerate(tokens):
        if not _ENV_VAR_RE.match(token):
            return env, tokens[i:]
        key, _, val = token.partition("=")
        env[key] = val
    return env, []


def _env_grant_prefix(env_vars: dict[str, str]) -> str:
    """Return a sorted ``KEY=*`` prefix string for grant keys and permission options."""
    return " ".join(f"{key}=*" for key in sorted(env_vars))


_CHAIN_OPS = frozenset({"||", "&&", "|", ";"})


def _parse_chain(command: str) -> list[tuple[str | None, str]]:
    """Split *command* on shell chain operators (``|``, ``&&``, ``||``, ``;``).

    Returns a list of ``(operator, raw_segment)`` pairs; the first operator is
    ``None``.  Each segment string is the **original unprocessed substring**
    (stripped of surrounding whitespace) rather than a ``shlex.join()``
    reconstruction.  Preserving raw substrings is important so that redirection
    operators like ``>`` and ``<`` are not re-quoted and remain detectable by
    ``_parse_redirections``.

    Raises ``ValueError`` if the command cannot be parsed (e.g. unclosed
    quotes) or has structural errors (leading/trailing chain operator).
    """
    # Validate via shlex to catch unclosed quotes — raises ValueError on bad input.
    # We do NOT use the shlex token list to check for trailing operators because
    # shlex strips quotes, so echo "&&" would produce a final token of "&&" and
    # be wrongly rejected.  Trailing-operator detection is done via scan state.
    lex = shlex.shlex(command, posix=True, punctuation_chars="|&;")
    lex.commenters = ""
    lex.whitespace_split = True
    list(lex)

    # Extract raw substrings with a lightweight quote-aware scan.  We only need
    # to track single/double-quote state and backslash escapes to know which
    # |/&&/||/; characters are real operators vs. quoted literals.
    segments: list[tuple[str | None, str]] = []
    current_op: str | None = None
    seg_start = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        c = command[i]
        if c == "\\" and not in_single:
            i += 2  # skip backslash-escaped character (unquoted or double-quoted)
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            two = command[i : i + 2]
            if two in ("&&", "||"):
                raw = command[seg_start:i].strip()
                if not raw and current_op is not None:
                    raise ValueError(
                        f"Empty segment between {current_op!r} and {two!r}"
                    )
                if raw:
                    segments.append((current_op, raw))
                current_op = two
                seg_start = i + 2
                i += 2
                continue
            elif c in ("|", ";"):
                raw = command[seg_start:i].strip()
                if not raw and current_op is not None:
                    raise ValueError(f"Empty segment between {current_op!r} and {c!r}")
                if raw:
                    segments.append((current_op, raw))
                current_op = c
                seg_start = i + 1
        i += 1

    raw = command[seg_start:].strip()
    if raw:
        segments.append((current_op, raw))
    elif current_op is not None:
        raise ValueError(f"Command ends with chain operator {current_op!r}")

    if segments and segments[0][0] is not None:
        raise ValueError(f"Command starts with chain operator {segments[0][0]!r}")

    return segments


def _chain_summary(segments: list[tuple[str | None, str]]) -> str:
    """Return a compact chain summary: executables and operators only.

    Example: segments from ``"cat foo | grep bar | awk '{print $2}'"``
    becomes ``"cat | grep | awk"``.
    """
    parts: list[str] = []
    for op, segment in segments:
        if op is not None:
            parts.append(op)
        try:
            tokens = shlex.split(segment)
        except ValueError:
            parts.append(segment)
            continue
        _, cmd_tokens = _split_env_vars(tokens)
        parts.append(cmd_tokens[0] if cmd_tokens else segment)
    return " ".join(parts)


def _tokenize_segment(text: str) -> list[tuple[str, int, int]]:
    """Tokenise *text* for redirection parsing.

    Returns a list of ``(value, raw_start, raw_end)`` tuples where
    *raw_start* and *raw_end* are character positions in *text* (exclusive).

    Rules mirror ``shlex(posix=True, punctuation_chars='<>', whitespace_split=True)``:
    - ``<`` and ``>`` are always emitted as (consecutive) punctuation tokens.
    - Single-quoted strings: literal content, no escapes.
    - Double-quoted strings: backslash escapes apply only to ``\\``, ``$``,
      `` ` ``, ``"`` and newline (POSIX rule); other backslashes are kept.
    - Backslash outside quotes escapes the next character.
    - Raises ``ValueError`` on unterminated quotes.
    """
    result: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        raw_start = i
        if c in "<>":
            while i < n and text[i] in "<>":
                i += 1
            result.append((text[raw_start:i], raw_start, i))
        else:
            chars: list[str] = []
            while i < n and text[i] not in " \t\r\n<>":
                ch = text[i]
                if ch == "'":
                    i += 1
                    while i < n and text[i] != "'":
                        chars.append(text[i])
                        i += 1
                    if i >= n:
                        raise ValueError("Unterminated single quote")
                    i += 1  # skip closing '
                elif ch == '"':
                    i += 1
                    while i < n:
                        if text[i] == "\\" and i + 1 < n and text[i + 1] in _DQ_ESCAPE:
                            # POSIX: backslash only escapes \, $, `, ", newline inside "…"
                            chars.append(text[i + 1])
                            i += 2
                        elif text[i] == '"':
                            i += 1
                            break
                        else:
                            chars.append(text[i])
                            i += 1
                    else:
                        raise ValueError("Unterminated double quote")
                elif ch == "\\" and i + 1 < n:
                    chars.append(text[i + 1])
                    i += 2
                else:
                    chars.append(ch)
                    i += 1
            result.append(("".join(chars), raw_start, i))
    return result


def _parse_redirections(segment: str) -> tuple[str, list[str]]:
    """Split a command segment into its base command and redirection list.

    Returns ``(command_part, redirections)`` where *command_part* is the
    ``shlex.join()`` of non-redirection tokens and *redirections* is a list
    of normalised strings such as ``"> output.txt"``, ``"2>&1"``, or
    ``">&2"``.

    ``<`` and ``>`` are always split out as separate tokens so that
    ``cmd>file``, ``cmd >file``, and ``cmd > file`` are all handled
    identically.

    An fd number (e.g. ``2`` in ``2>&1``) is only absorbed as an fd prefix
    when it was **immediately adjacent** (no whitespace) to the operator in the
    original string.  This is verified via the raw token positions returned by
    ``_tokenize_segment``, so ``"echo 2 > file"`` correctly leaves ``2`` as a
    command argument while ``"echo 2>file"`` correctly treats ``2`` as an fd.

    Recognised redirection forms:
    - ``>``, ``>>``, ``N>``, ``N>>`` followed by a filename.
    - ``<`` followed by a filename.
    - ``N>&M`` or ``>&M`` (self-contained, no filename), e.g. ``2>&1``.

    Quoted or backslash-escaped ``>``/``<`` tokens (e.g. ``echo '>' foo``,
    ``echo \\> foo``) are detected via the raw-position guard
    ``segment[raw_start:raw_end] == val`` and left as command arguments.
    """
    try:
        tok_list = _tokenize_segment(segment)
    except ValueError:
        return segment, []

    # tok_list entries: (value, raw_start, raw_end)
    # cmd_token_ends and cmd_token_starts track the raw extents of each entry
    # in cmd_tokens so that fd-absorption can verify adjacency and quoting.
    cmd_tokens: list[str] = []
    cmd_token_ends: list[int] = []
    cmd_token_starts: list[int] = []
    redirs: list[str] = []
    i = 0
    while i < len(tok_list):
        val, raw_start, raw_end = tok_list[i]
        if _REDIR_OP_RE.match(val) and segment[raw_start:raw_end] == val:
            # Guard: only treat the token as an operator when it is an unquoted,
            # unescaped bare operator in the original string (raw slice == value).
            # echo '>' foo and echo \> foo must not produce a redirection entry.
            op = val
            # Absorb the preceding digit as an fd-prefix ONLY when:
            #   1. it was immediately adjacent (digit's raw_end == op's raw_start), AND
            #   2. the raw slice in the original string is all digits — no quoting,
            #      no backslash escapes.  Shell requires a bare integer IO number.
            # e.g. "2>&1"/"2>file" → absorb; "2 > file"/"'2'>file"/"\2>file" → skip.
            if (
                _REDIR_BARE_OP_RE.match(op)
                and cmd_tokens
                and _REDIR_FD_RE.match(cmd_tokens[-1])
                and cmd_token_ends[-1] == raw_start
                and segment[cmd_token_starts[-1] : cmd_token_ends[-1]].isdigit()
            ):
                op = cmd_tokens.pop() + op  # e.g. "2" + ">" → "2>"
                cmd_token_ends.pop()
                cmd_token_starts.pop()
            next_i = i + 1
            if next_i < len(tok_list):
                next_val, next_raw_start, next_raw_end = tok_list[next_i]
            else:
                next_val, next_raw_start, next_raw_end = None, 0, 0
            if (
                next_val
                and _REDIR_FD_TARGET_RE.match(next_val)
                and raw_end == next_raw_start  # must be immediately adjacent to op
                and segment[next_raw_start:next_raw_end]
                == next_val  # unquoted/unescaped
            ):
                # e.g. "&1" as a single token → "2>&1"
                redirs.append(op + next_val)
                i += 2
            elif (
                next_val == "&"
                and raw_end == next_raw_start  # & must be adjacent to the op
                and segment[next_raw_start:next_raw_end] == "&"  # unquoted &
                and next_i + 1 < len(tok_list)
                and _REDIR_FD_RE.match(tok_list[next_i + 1][0])
            ):
                # "&" and the digit as separate tokens (e.g. "> &" written
                # without adjacency would fail the raw_end check above).
                redirs.append(op + "&" + tok_list[next_i + 1][0])
                i += 3
            elif next_val is not None:
                # Use the original raw slice as the target so that the stored
                # redir string preserves quoting.  This prevents an exact grant
                # for "> '$(id)'" from also matching "> $(id)" (unquoted form
                # that would undergo shell expansion at execution time).
                target_raw = segment[next_raw_start:next_raw_end]
                redirs.append(f"{op} {target_raw}")
                i += 2
            else:
                # Dangling operator — treat as a regular argument.
                cmd_tokens.append(op)
                cmd_token_ends.append(raw_end)
                cmd_token_starts.append(raw_start)
                i += 1
        else:
            cmd_tokens.append(val)
            cmd_token_ends.append(raw_end)
            cmd_token_starts.append(raw_start)
            i += 1

    return shlex.join(cmd_tokens) if cmd_tokens else "", redirs


def _redir_pattern_to_regex(pattern: str) -> str:
    """Translate an fnmatch-style *pattern* to a regex where wildcards don't cross path separators.

    Supports the same constructs as ``fnmatch`` (``*``, ``?``, ``[…]``) but
    ``*`` and ``?`` translate to ``[^/\\\\]*`` and ``[^/\\\\]`` respectively so
    that they cannot match ``/`` or ``\\``.  ``[…]`` character classes are
    passed through unchanged.
    """
    i = 0
    n = len(pattern)
    parts: list[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            parts.append(r"[^/\\]*")
            i += 1
        elif c == "?":
            parts.append(r"[^/\\]")
            i += 1
        elif c == "[":
            # Locate the matching ']', respecting leading '!' and ']' as first char.
            j = i + 1
            if j < n and pattern[j] == "!":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                parts.append(re.escape(c))  # unclosed '[' treated as literal
                i += 1
            else:
                char_class = pattern[i : j + 1]
                # fnmatch uses '[!' for negation; regex uses '[^'
                if char_class.startswith("[!"):
                    char_class = "[^" + char_class[2:]
                parts.append(char_class)
                i = j + 1
        else:
            parts.append(re.escape(c))
            i += 1
    return "".join(parts)


def _redir_pattern_match(redir: str, pattern: str) -> bool:
    """Return True if *redir* matches *pattern* with wildcards not crossing path separators.

    Supports ``fnmatch``-style ``*``, ``?``, and ``[…]`` constructs, but
    ``*`` and ``?`` are bounded by ``/`` and ``\\`` so that ``"> *"`` cannot
    match ``"> /etc/passwd"`` or ``"> C:\\\\Windows\\\\..."``.

    On Windows, ``os.path.normcase`` is applied to both operands so matching
    is case-insensitive and slash-normalised, consistent with ``fnmatch.fnmatch``.
    """
    redir = os.path.normcase(redir)
    pattern = os.path.normcase(pattern)
    return bool(re.fullmatch(_redir_pattern_to_regex(pattern), redir))


def _grant_key(command: str) -> str:
    """Return the normalised command key for grant matching.

    Leading env var assignments are included by variable name only, not by
    value. This makes grants sensitive to which leading env vars are present
    (for example, adding or removing ``PATH=...`` or ``LD_PRELOAD=...`` changes
    the key), but changing the value of an already-present env var does not.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command.strip()
    env_vars, cmd_tokens = _split_env_vars(tokens)
    cmd_key = shlex.join(cmd_tokens)
    if not env_vars:
        return cmd_key
    env_key = _env_grant_prefix(env_vars)
    return f"{env_key} {cmd_key}" if cmd_key else env_key


class BashTool(Tool):
    NAME = "bash"
    DESCRIPTION = "Run an arbitrary shell command on the client computer."
    PERMISSION_REQUIRED = True
    DISABLED_BY_DEFAULT = True

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        permission_required: bool,
        name: str,
        description: str,
    ) -> None:
        super().__init__(
            workspace, permission_manager, permission_required, name, description
        )
        self._exact_grants: set[str] = set()
        self._pattern_grants: list[str] = []

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        self._exact_grants.clear()
        self._pattern_grants.clear()
        logger.debug("bash: session grants cleared")

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def _command_is_granted(self, cmd_part: str) -> bool:
        """Return True if *cmd_part* has an existing exact or pattern grant."""
        normalized = _grant_key(cmd_part)
        if normalized in self._exact_grants:
            return True
        return any(
            fnmatch.fnmatch(normalized, p) or fnmatch.fnmatch(normalized + " ", p)
            for p in self._pattern_grants
        )

    def _redir_is_granted(self, redir: str) -> bool:
        """Return True if *redir* has an existing exact or pattern grant.

        Wildcard pattern grants are never applied when the redirection target
        contains shell metacharacters (``$``, backtick, ``(``, etc.) that could
        trigger command substitution when the command runs under ``shell=True``.
        Such targets require a fresh explicit approval regardless of stored grants.
        """
        if redir in self._exact_grants:
            return True
        if _REDIR_SHELL_META_RE.search(redir):
            return False
        return any(_redir_pattern_match(redir, p) for p in self._pattern_grants)

    def request_permission(self, action: str, **kwargs: Any) -> tuple[bool, str]:
        if not self.permission_required:
            return True, ""
        cmd = kwargs.get("command", "")

        if _has_heredoc(cmd):
            # Heredoc: bypass chain parsing entirely and treat the whole command
            # as one-time permission with no grant storage.
            #
            # We cannot safely run _parse_chain on a heredoc command because the
            # body may contain operator characters (|, &&) or unmatched quotes that
            # the parser would mis-classify as real chain operators.  Treating the
            # whole command as one-time is the safest approach; it means that even
            # non-heredoc segments in a mixed chain (e.g. "echo hi | cat <<EOF\n…")
            # lose their per-segment grant granularity, but that trade-off avoids
            # incorrect permission splits.
            #
            # extra_permission_options() returns ["always"] for heredoc, which
            # intercepts the universal always-choice so that PermissionManager
            # cannot create a session-wide grant.  We discard the choice on
            # success so that on_permission_granted() is never invoked and no
            # grant is stored for heredoc commands.
            #
            # _request_permission_as("bash_heredoc", …) is used instead of
            # super().request_permission() so that a pre-existing tool-wide
            # always-grant for "bash" cannot silently bypass prompting.  Heredocs
            # must always prompt because their body content is dynamic and cannot
            # be granted in advance.
            heredoc_note = (
                ' (heredoc: one-time approval only — "always" will not be saved)'
            )
            allowed, hd_choice = self._request_permission_as(
                "bash_heredoc", action + heredoc_note, command=cmd
            )
            return allowed, hd_choice if not allowed else ""

        try:
            segments = _parse_chain(cmd) if cmd else []
        except ValueError:
            segments = []

        if not segments:
            return super().request_permission(action, **kwargs)

        # Route all commands (single or chained) through the unified flow so
        # that redirection tokens within single segments are also checked.
        return self._request_chain_permission(segments, action=action)

    def _request_chain_permission(
        self,
        segments: list[tuple[str | None, str]],
        action: str = "",
    ) -> tuple[bool, str]:
        """Request per-segment (and per-redirection) permission for a command.

        Always-type grants are recorded immediately so they survive even when
        a later segment is denied.  *action* is used verbatim as the question
        text for plain single-segment commands (no chain, no redirections).

        Each *segment* string is the raw original substring from ``_parse_chain``
        (not a ``shlex.join`` reconstruction), so ``_parse_redirections`` can
        detect embedded operators like ``>`` and ``<`` reliably.
        """
        summary = _chain_summary(segments)
        is_single = len(segments) == 1

        for op, segment in segments:
            cmd_part, redirs = _parse_redirections(segment)

            # --- command permission ---
            if cmd_part:
                if self._command_is_granted(cmd_part):
                    logger.debug("bash: segment %r granted (cache)", cmd_part)
                else:
                    op_str = f" {op}" if op else ""
                    if is_single and not redirs:
                        # Preserve the caller-supplied action text.
                        cmd_action = action
                    elif is_single:
                        cmd_action = f"Run: {cmd_part}"
                    else:
                        cmd_action = f"Chain: {summary}\nRun{op_str}: {cmd_part}"
                    allowed, choice = super().request_permission(
                        cmd_action, command=cmd_part
                    )
                    # Store always-grants before checking later segments/redirs.
                    if allowed and choice:
                        self.on_permission_granted(choice, command=cmd_part)
                    if not allowed:
                        return False, choice

            # --- redirection permissions ---
            for redir in redirs:
                if self._redir_is_granted(redir):
                    logger.debug("bash: redirect %r granted (cache)", redir)
                    continue
                op_str = f" {op}" if op else ""
                if is_single:
                    redir_action = f"Redirect: {redir}"
                else:
                    redir_action = f"Chain: {summary}\nRedirect{op_str}: {redir}"
                allowed, choice = super().request_permission(
                    redir_action, redirection=redir
                )
                if allowed and choice:
                    self.on_permission_granted(choice, redirection=redir)
                if not allowed:
                    return False, choice

        return True, ""

    def extra_permission_options(self, **kwargs: Any) -> list[str]:
        """Return extra permission options.

        For redirections with a filename: ``["always", "always: <op> <dir>/*"]``.
        For self-contained redirections (e.g. ``2>&1``): ``["always"]``.

        For commands:
        - Normal case (2+ tokens): ``["always", "always: <exe> <leading_args> *"]``.
        - Single-token command:    ``["always", "always: <exe> *"]``.
        - Unparseable / empty:     ``["always"]``.

        Including ``"always"`` intercepts the universal always-choice before
        PermissionManager records it as a tool-wide grant, so that
        ``on_permission_granted`` can store a command-specific exact grant instead.
        """
        redir = kwargs.get("redirection", "")
        if redir:
            return self._redir_extra_options(redir)

        cmd = kwargs.get("command", "")
        if not cmd:
            return ["always"]
        if _has_heredoc(cmd):
            # Heredoc content is dynamic — no persistent grant is meaningful.
            # "always" intercepts the universal always-choice so that
            # PermissionManager cannot create a session-wide bash grant.
            return ["always"]
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return ["always"]
        if not tokens:
            return ["always"]
        env_vars, cmd_tokens = _split_env_vars(tokens)
        if not cmd_tokens:
            return ["always"]
        env_prefix = _env_grant_prefix(env_vars)
        if len(cmd_tokens) == 1:
            exe_part = f"{env_prefix} {cmd_tokens[0]}" if env_prefix else cmd_tokens[0]
            return ["always", f"always: {exe_part} *"]
        leading = shlex.join(cmd_tokens[:-1])
        cmd_part = f"{env_prefix} {leading}" if env_prefix else leading
        return ["always", f"always: {cmd_part} *"]

    def _redir_extra_options(self, redir: str) -> list[str]:
        """Return extra permission options for a redirection string."""
        parts = redir.split(None, 1)
        if len(parts) < 2:
            # Self-contained (e.g. "2>&1") — exact match only.
            return ["always"]
        op, filename = parts
        # If the filename starts with a shell quote character, any dirname-derived
        # wildcard pattern would have an unbalanced quote (e.g. "> './docs/*").
        # Fall back to exact-match only for quoted filenames.
        if filename[:1] in ("'", '"'):
            return ["always"]
        # Wildcard patterns are never matched against targets that contain shell
        # metacharacters (see _redir_is_granted).  Offering a wildcard option for
        # such a target would present a choice that can never be exercised.
        if _REDIR_SHELL_META_RE.search(filename):
            return ["always"]
        # Use the last occurrence of either '/' or '\' as the separator so that
        # the parent matches _redir_pattern_match()'s separator rules regardless
        # of the host OS (os.path.dirname ignores '\' on POSIX).
        last_sep = max(filename.rfind("/"), filename.rfind("\\"))
        if last_sep < 0:
            pattern = f"{op} *"
        else:
            sep_char = filename[last_sep]
            parent = filename[:last_sep]  # empty string when last_sep == 0 (root)
            # Root-level file (e.g. "/out.txt"): emit "> /*" not "> //*".
            pattern = f"{op} {parent}{sep_char}*" if parent else f"{op} {sep_char}*"
        return ["always", f"always: {pattern}"]

    def on_permission_granted(self, choice: str, **kwargs: Any) -> None:
        redir = kwargs.get("redirection", "")
        if redir:
            if choice == "always":
                self._exact_grants.add(redir)
                logger.info("bash: exact redirect grant stored for %r", redir)
            elif choice.startswith("always: ") and choice.endswith("*"):
                pattern = choice[len("always: ") :]
                if pattern not in self._pattern_grants:
                    self._pattern_grants.append(pattern)
                    logger.info("bash: redirect pattern grant stored: %r", pattern)
            return

        cmd = kwargs.get("command", "")
        if not cmd:
            return
        normalized = _grant_key(cmd)
        if not normalized:
            return
        if choice == "always":
            self._exact_grants.add(normalized)
            logger.info("bash: exact grant stored for %r", normalized)
        elif choice.startswith("always: ") and choice.endswith("*"):
            pattern = choice[len("always: ") :]
            if pattern not in self._pattern_grants:
                self._pattern_grants.append(pattern)
                logger.info("bash: pattern grant stored: %r", pattern)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def execute_log(self, **kwargs: Any) -> str | None:
        cmd: str = kwargs.get("command", "")
        if not cmd:
            return "<empty command>"
        try:
            segments = _parse_chain(cmd)
        except ValueError:
            return "<unparseable command>"
        if len(segments) > 1:
            # Show raw command for chains — shlex.join would quote the operators.
            return cmd if len(cmd) <= 60 else f"{cmd[:57]}..."
        if not segments:
            return "<empty command>"
        _, segment = segments[0]
        _, redirs = _parse_redirections(segment)
        if redirs:
            # Show raw command when redirections are present.
            return cmd if len(cmd) <= 60 else f"{cmd[:57]}..."
        # Single command without redirections — show canonical form.
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return "<unparseable command>"
        if not tokens:
            return "<empty command>"
        _, cmd_tokens = _split_env_vars(tokens)
        if not cmd_tokens:
            return "<empty command>"
        summary = shlex.join(tokens)
        return summary if len(summary) <= 60 else f"{summary[:57]}..."

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Run a shell command on the client computer and return its output. "
                "Supports piping, boolean operators (&&, ||), and semicolons; "
                "each segment requires separate user approval."
            ),
            arguments=[
                ToolArgument(
                    name="command",
                    description="The shell command to run (e.g. 'ls -la ./src').",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="capture",
                    description=(
                        "Which output stream(s) to capture. "
                        "'stdout' (default) captures stdout only. "
                        "'stderr' captures stderr only. "
                        "'interleaved' merges stderr into stdout. "
                        "'separate' returns stdout and stderr as separate fields."
                    ),
                    argument_type="string",
                    required=False,
                    enum=list(_CAPTURE_MODES),
                ),
                ToolArgument(
                    name="max_output_chars",
                    description=(
                        "Maximum number of characters to return from captured output "
                        f"(default {_DEFAULT_MAX_OUTPUT_CHARS}). Output beyond this "
                        "limit is truncated."
                    ),
                    argument_type="integer",
                    required=False,
                    minimum=1,
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        command: str,
        capture: str = "stdout",
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    ) -> dict:
        if capture not in _CAPTURE_MODES:
            return self._err(
                "invalid_arguments",
                f"Invalid capture mode {capture!r}. Must be one of: "
                + ", ".join(_CAPTURE_MODES),
                400,
            )
        if max_output_chars < 1:
            return self._err(
                "invalid_arguments",
                f"max_output_chars must be >= 1, got {max_output_chars}.",
                400,
            )

        # Detect chain operators / redirections to choose execution strategy.
        # Heredoc syntax (<<MARKER) must be detected first: the heredoc body can
        # contain unmatched quotes or operator characters that shlex would reject,
        # so we bypass _parse_chain entirely for heredoc commands.
        has_heredoc = _has_heredoc(command)
        segments: list[tuple[str | None, str]] = []
        is_chain = False
        has_redirections = False
        if not has_heredoc:
            try:
                segments = _parse_chain(command)
            except ValueError as exc:
                logger.debug("bash: shlex parse failed: %s", exc)
                return self._err(
                    "invalid_command", f"Failed to parse command: {exc}", 400
                )
            is_chain = len(segments) > 1
            # Redirections in a single-segment command also require shell semantics.
            # Use the original command string for redirection detection; _parse_chain
            # returns raw substrings, but the single-segment fast-path re-parses the
            # full original string so that redirection operators are always detectable.
            has_redirections = not is_chain and bool(_parse_redirections(command)[1])

        # Build stream kwargs for subprocess based on capture mode.
        if capture == "interleaved":
            stream_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            }
        elif capture == "stderr":
            stream_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
        elif capture == "separate":
            stream_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        else:
            # stdout only — stderr captured for error reporting on non-zero exit
            stream_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

        if is_chain or has_redirections or has_heredoc:
            logger.debug(
                "bash: running command via shell (chain=%s, redirs=%s, heredoc=%s)",
                is_chain,
                has_redirections,
                has_heredoc,
            )
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    **stream_kwargs,
                    text=True,
                    timeout=_DEFAULT_TIMEOUT,
                    cwd=self._workspace.root,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "bash: shell command timed out after %ds", _DEFAULT_TIMEOUT
                )
                return self._err(
                    "execution_error",
                    f"Command timed out after {_DEFAULT_TIMEOUT} seconds.",
                    408,
                )
            except Exception as exc:
                logger.exception("bash: unexpected error running shell command")
                return self._err("execution_error", str(exc), 500)
        else:
            if not segments:
                return self._err("invalid_command", "Command is empty.", 400)
            args = shlex.split(command)
            if not args:
                return self._err("invalid_command", "Command is empty.", 400)
            env_vars, cmd_args = _split_env_vars(args)
            if not cmd_args:
                return self._err(
                    "invalid_command",
                    "Command is empty after stripping environment variables. "
                    "Expected an executable after env assignments, like `A=1 ls`.",
                    400,
                )
            logger.debug("bash: running %r (%d arg(s))", cmd_args[0], len(cmd_args) - 1)
            subprocess_env = {**os.environ, **env_vars} if env_vars else None
            try:
                proc = subprocess.run(
                    cmd_args,
                    **stream_kwargs,
                    text=True,
                    timeout=_DEFAULT_TIMEOUT,
                    cwd=self._workspace.root,
                    stdin=subprocess.DEVNULL,
                    env=subprocess_env,
                )
            except FileNotFoundError:
                logger.debug("bash: executable not found: %r", cmd_args[0])
                return self._err(
                    "execution_error", f"Command not found: {cmd_args[0]}", 400
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "bash: %r timed out after %ds", cmd_args[0], _DEFAULT_TIMEOUT
                )
                return self._err(
                    "execution_error",
                    f"Command timed out after {_DEFAULT_TIMEOUT} seconds.",
                    408,
                )
            except Exception as exc:
                logger.exception("bash: unexpected error running %r", cmd_args[0])
                return self._err("execution_error", str(exc), 500)

        if proc.returncode != 0:
            logger.debug("bash: command exited with status %d", proc.returncode)
            message = f"Command exited with status {proc.returncode}."
            # interleaved merges stderr into stdout; proc.stderr is None in that mode
            raw_error = (proc.stdout if capture == "interleaved" else proc.stderr) or ""
            if raw_error:
                error_output, _ = _truncate(raw_error.strip(), max_output_chars)
                message = f"{message} {error_output}"
            return self._err("execution_error", message, 400)

        if capture == "separate":
            stdout_text, stdout_truncated = _truncate(
                proc.stdout or "", max_output_chars
            )
            stderr_text, stderr_truncated = _truncate(
                proc.stderr or "", max_output_chars
            )
            data: dict[str, Any] = {"stdout": stdout_text, "stderr": stderr_text}
            if stdout_truncated or stderr_truncated:
                data["warning"] = f"Output truncated at {max_output_chars} characters"
        else:
            raw = (proc.stderr or "") if capture == "stderr" else (proc.stdout or "")
            logger.debug("bash: command succeeded, output=%d chars", len(raw))
            text, truncated = _truncate(raw, max_output_chars)
            data = {"output": text}
            if truncated:
                data["warning"] = f"Output truncated at {max_output_chars} characters"

        return self._ok(data)
