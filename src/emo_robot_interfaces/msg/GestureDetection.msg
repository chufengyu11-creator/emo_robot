uint8 GESTURE_NONE=0
uint8 GESTURE_WAVE=1
uint8 GESTURE_COME=2

std_msgs/Header header
int32 track_id
uint8 gesture
float32 confidence
float32 center_x
float32 center_y
bool valid
