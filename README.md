# lms-cli
A LMStudio CLI wrapper capable of reading and editing files in a local folder

## Structure

LMS-cli consists of two components: An interactive CLI-based user interface and a REST
API server that actually handles the conversation with LM Studio, keeps track of
sessions, as well as runs the various tools for manipulating files etc.

## REST API Backend

Running on TCP/IP port `13518` by default, but this can be overridden using an argument
or the configuration file `.lms-cli-api-config.json`.

The configuration file also contains information about the LM Studio server the backend
should communicate with etc.

```json
{
    "bind_address": "0.0.0.0",
    "bind_port": null,
    "lms_server": "http://192.168.1.8:1234/v1",
    "lms_model": "mistralai/devstral-small-2-2512",
    "context_window": 4096
}
```

## CLI UI tool

An interactive prompt using the REST API backend to communicate with LM Studio and call
tools.  Defaults to starting the rest backend itself as a subprocess, but can be told
to connect to an existing server using CLI arguments.
