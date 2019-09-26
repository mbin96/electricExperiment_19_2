#pragma once
#pragma warning(disable : 4996)
#define DEBUG_

//define constant
#define CHANNEL 3
#define FILTER_H 3
#define FILTER_W 3
#define WINDOW_SIZE 5
#define PI 3.1415


#define THRESHOLD 60
float CONT_k = 0.04;

//struct definition
struct pixel {
    //gradiant y
    float H;
    //gradiant x
    float W;
    //raw edge weight or true/false
    float edge;
    //0~8
    float phase;
    float magnitude;
    float hog[9];
}typedef pixel;

//gradient filter definition
int filterX[9] = {
    -1, -1, -1,
    0,  0,  0,
    1,  1,  1
};
int filterY[9] = {
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
};
