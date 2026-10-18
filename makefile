CC = clang
CFLAGS = -Wall -Wextra -Werror -O2
TARGET = main
SRC = main.c

# Default target: compile AND run
all: run

# Build the program
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Run the program (depends on build)
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET)
