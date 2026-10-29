extends Node

## Headless training host. Listens for the trainer and answers one JSON message per line.
## Strict request-reply order.

const PORT: int = 11008

var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
## Bytes recieved but not yet ending in a newline.
var _buffer := PackedByteArray()

func _ready() -> void:
    var err := _server.listen(PORT, "127.0.0.1")
    if err != OK:
        push_error("host: cannot listen on port %d (error %d)" % [PORT, err])
        get_tree().quit(1)
        return
    print("host: listening on %d" % PORT)

func _physics_process(_delta: float) -> void:
    if _peer == null:
        if _server.is_connection_available():
            _peer = _server.take_connection()
            _peer.set_no_delay(true)
            print("host: trainer connected")
        return
    
    _peer.poll()
    if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
        print("host: trainer disconnected")
        get_tree().quit()
        return
    
    var available := _peer.get_available_bytes()
    if available > 0:
        var result: Array = _peer.get_data(available)
        if result[0] != OK:
            push_error("host: read failed (error %d)" % result[0])
            return
        _buffer.append_array(result[1])
    
    var newline := _buffer.find(10)
    while newline >= 0:
        var line := _buffer.slice(0, newline).get_string_from_utf8()
        _buffer = _buffer.slice(newline + 1)
        _handle(line)
        newline = _buffer.find(10)


func _handle(line: String) -> void:
    var message = JSON.parse_string(line)
    if typeof(message) != TYPE_DICTIONARY:
        _send({"type": "error", "reason": "not a JSON object: %s" % line})
        return
    
    match message.get("type", ""):
        "hello": _send({"type": "hello", "obs_size": 0, "act_size": NeuralPilot.ACT_SIZE})
        _: _send({"type": "error", "reason": "unknown message type '%s'" % message.get("type", "")})

func _send(message: Dictionary) -> void:
    _peer.put_data((JSON.stringify(message) + "\n").to_utf8_buffer())
            
