#include "base/base_all.h"
#include <base/base_arena.h>
#include <base/base_defs.h>
#include <base/base_strings.h>

int main(void) {
	Arena arena = arena_create(MB(5));
	String some_string = string_from_cstr("this is some string");
	char *cstr = string_to_cstr(&arena, some_string);
}
