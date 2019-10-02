echo "opencv docker - $1 build start" 
g++ -ggdb $1.cpp -o $1 `pkg-config --cflags --libs opencv`
echo "opencv docker - $1 excu"
./$1