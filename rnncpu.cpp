#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std; 

void mm_simple(float *A, float *B, float *C, int m, int n, int p)
{
    float sum = 0.0f;
    for (int i = 0; i < m; i++)
    {
        for (int k = 0; k < p; k++)
        {
            sum = 0.0f;
            for (int j = 0; j < n; j++)
            {
                sum += A[n*i+j] * B[p*j+k];
            }
            C[i*p+k] = sum;
        }
        
    }
}

void simple_add_tensor(float* A, float* B, float* C, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            C[i*y+j] = A[i*y+j] + B[i*y+j];
        }
        
    }
}

void tanh_activation(float* arr, int m, int n)
{
    for (int i = 0; i < m*n; i++)
    {
        arr[i] = tanh(arr[i]);
    }
}

void printArray(float* arr, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            cout << arr[i*y+j] << "\t";
        }
        cout << endl;
    }
}

void initializerandarray(float* A, int m, int n)
{
    float x = 5.0;
    for (int i = 0; i < m*n; i++)
    {
        float r3 = -5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10)));
        A[i] = r3;
    }
}


void rnn_forward_step(float* x, float* h, float* y, float* U, float* V, float* W)
{
    // Step 1: Calculate x(t)*U and h(t-1)*W
    float xt_times_U[70];
    float prevht_times_W[70];
    mm_simple(x,U,xt_times_U,10,1,7);
    mm_simple(h,W,prevht_times_W,10,7,7);
    simple_add_tensor(xt_times_U,prevht_times_W,h,10,7);    //elementwise addition
    tanh_activation(h,10,7);    //elementwise tanh
    mm_simple(V,h,y,1,10,7);
}

int main()
{
    srand((unsigned int)time(NULL));
    /*
        Hard coded dimensions
            xt = 10x1
            ht = 10x7
            yt = 1x7
            U = 1x7
            V = 1x10
            W = 7x7
    */
    

    // Create a sequence of xt and yt
    // pass it thru the forward pass how many ever times we wanna do that
    float xt[10];
    float ht[70];
    float yt[7];
    float U[7];
    float V[10];
    float W[49];
    initializerandarray(xt,10,1);
    initializerandarray(ht,10,7);
    initializerandarray(yt,1,7);
    initializerandarray(U,1,7);
    initializerandarray(V,1,10);
    initializerandarray(W,7,7);
    cout << "xt" << endl;
    printArray(xt,10,1);
    cout << endl;
    cout << "ht" << endl;
    printArray(ht,10,7);
    cout << endl;
    cout << "yt" << endl;
    printArray(yt,1,7);
    cout << endl;
    cout << "U" << endl;
    printArray(U,1,7);
    cout << endl;
    cout << "V" << endl;
    printArray(V,1,10);
    cout << endl;
    cout << "W" << endl;
    printArray(W,7,7);
    cout << endl;
    rnn_forward_step(xt,ht,yt,U,V,W);
    cout << "ht" << endl;
    printArray(ht,10,7);
    cout << endl;
    cout << "yt" << endl;
    printArray(yt,1,7);
}