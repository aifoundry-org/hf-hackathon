/* Null kernel — measures boot overhead only. */
#include "yolo_common.h"
int main(uintptr_t arg_area) {
    (void)arg_area;
    uint32_t hid = get_hart_id();
    if (hid != 0) return 0;
    return 0;
}