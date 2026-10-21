#include "base/base_minimal.h"
#include "small_tui/tui.h"
#include <base/base.h>

// Small nn to identiy bird call signals

#define N 16

structdef(Drawing){
	bool pixels[N][N];
}; structdef_end(Drawing)

Drawing paint(void);

int main(void){
    for (int i = 0; i < 1; i++){
    	Drawing character = paint();
    	printf("int pixels[%d][%d] = {\n", N, N);
        for (int y = 0; y < N; y++) {
            printf("    {");
            for (int x = 0; x < N; x++)
                printf("%d%s", character.pixels[y][x] ? 1 : 0, x < N - 1 ? ", " : "");
            printf("}%s\n", y < N - 1 ? "," : "");
        }
    }
}


Drawing paint(void) {
    Drawing drawing = {0};
    bool mode = true;
    int  last_x = -1, last_y = -1;
    uint64_t black = COLOR(C_RGB(0, 0, 0), C_RGB(0, 0, 0));
    uint64_t white = COLOR(C_RGB(5, 5, 5), C_RGB(5, 5, 5));
    uint64_t ui    = COLOR(C_BRIGHT_WHITE, C_SHADE(3), C_CYAN);
    while (tui_run(0)) {
        fill(0, 0, A, A, ui);
        scope (A, A, N * 2 + 2, N + 6) {
            panel("Mini Paint", 0, 0, ~0, ~0, ui);
            scope (1, 1, N * 2, N) {
                struct rect r = abs_xywh(0, 0, N * 2, N);   // screen position
                bool press = is_event_key(MOUSE_EVENT, LEFT_BUTTON);
                bool drag  = is_event_key(MOUSE_EVENT, LEFT_DRAG);
                if ((press || drag) && is_mouse_over(r)) {
                    int tx = (tui.event.x - r.x) / 2;
                    int ty = tui.event.y - r.y;
                    if (press) {
                        mode   = !drawing.pixels[ty][tx];
                        last_x = tx;
                        last_y = ty;
                    }
                    int x0 = last_x < 0 ? tx : last_x;
                    int y0 = last_y < 0 ? ty : last_y;
                    int steps = TUI_MAX(abs(tx - x0), abs(ty - y0));
                    for (int i = 0; i <= steps; i++) {
                        int px = steps ? x0 + (tx - x0) * i / steps : tx;
                        int py = steps ? y0 + (ty - y0) * i / steps : ty;
                        drawing.pixels[py][px] = mode;
                    }
                    last_x = tx;
                    last_y = ty;
                } else if (drag) {
                    last_x = last_y = -1;
                }

                for (int y = 0; y < N; y++)
                    for (int x = 0; x < N; x++)
                        fill(x * 2, y, 2, 1, drawing.pixels[y][x] ? white : black);
            }
            if (button("Clear", 1, ~1, A, A, ui))
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        drawing.pixels[i][j] = false;
            if (button("Quit", ~1, ~1, A, A, ui))
                tui_quit();
        }
        if (is_key_press('q'))
            tui_quit();
    }
    return drawing;
}
