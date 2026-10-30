CC = clang
CFLAGS = -Wall -Wextra -Werror -O2
TARGET = birds
SRC = bethesda_bird_classifier.c

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
