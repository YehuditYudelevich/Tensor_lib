CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Werror -pedantic
TARGET = test_tensor.exe
OBJS = tensor.o tensor_view.o tensor_broadcast.o utils.o test_tensor.o

.PHONY: all test clean

all: test

test: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

tensor.o: tensor.c tensor.h utils.h error_handle.h
	$(CC) $(CFLAGS) -c tensor.c -o tensor.o

tensor_view.o: tensor_view.c tensor_view.h tensor_internal.h tensor.h utils.h error_handle.h
	$(CC) $(CFLAGS) -c tensor_view.c -o tensor_view.o

tensor_broadcast.o: tensor_broadcast.c tensor_broadcast.h tensor.h error_handle.h
	$(CC) $(CFLAGS) -c tensor_broadcast.c -o tensor_broadcast.o

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

test_tensor.o: test_tensor.c tensor.h tensor_view.h tensor_broadcast.h utils.h error_handle.h
	$(CC) $(CFLAGS) -c test_tensor.c -o test_tensor.o

clean:
	rm -f $(TARGET) *.o
