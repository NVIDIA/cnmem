#include <gtest/gtest.h>
#include <cnmem.h>
#include <stdint.h>
#include <fstream>

static std::size_t getFreeMemory() {
    cudaFree(0);
    std::size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    return freeMem;
}

class CnmemTest : public ::testing::Test {
    /// We determine the amount of free memory.
    std::size_t mFreeMem;
    
protected:
    /// Do we test memory leaks.
    bool mTestLeaks;
    /// Do we skip finalization.
    bool mFinalize;
    
public:
    /// Ctor.
    CnmemTest() : mFreeMem(getFreeMemory()), mTestLeaks(true), mFinalize(true) {}
    /// Tear down the test.
    void TearDown();
};

void CnmemTest::TearDown() {
    if( mFinalize ) {
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFinalize()); 
    }
    if( mTestLeaks ) {
        ASSERT_EQ(mFreeMem, getFreeMemory());
    }
    cudaDeviceReset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void tinyKernel(T* d_a, int numElem)
{
    int ind = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(ind >= numElem)
        return;
    d_a[ind] += 1;
}


struct _24ByteStruct
{
    double a;
    double c;
    double b;

    __host__ __device__
    void operator +=(int other)
    {
        a += other;
        b += other;
        c += other;
    }

    __host__ __device__
    void operator =(int other)
    {
        a = other;
        b = other;
        c = other;
    }
};

template<typename T, int expectedSize>
void testAlign()
{
    const int numElem = 200;
    const int size = numElem*sizeof(T);
    T* cpuData = new T[numElem];
    for(int i = 0; i < numElem; i++)
        cpuData[i] = i;

    ASSERT_EQ(expectedSize, sizeof(T));

    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.numStreams = 2;
    device.streams = streams;
    //intentonally misallign, but could be from calculation based on gpu size
    size_t streamSizes[] = { size*2 + sizeof(T) - 1, size*2 + sizeof(T) - 1 };
    device.streamSizes = streamSizes;

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT));
    T *ptr0, *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc((void**)&ptr0, size, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc((void**)&ptr1, size, streams[1]));

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(ptr0, cpuData, size, cudaMemcpyHostToDevice, streams[0]));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(ptr1, cpuData, size, cudaMemcpyHostToDevice, streams[1]));

    //force read and write from ptr0,1
    tinyKernel<<<numElem, 1, 0, streams[0]>>>(ptr0, numElem);
    tinyKernel<<<numElem, 1, 0, streams[1]>>>(ptr1, numElem);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[1]));

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[1])); 
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));
    
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[1]));

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

TEST_F(CnmemTest, alignment8) {
    testAlign<char, 1>();
}

TEST_F(CnmemTest, alignment16) {
    testAlign<short, 2>();
}

TEST_F(CnmemTest, alignment32) {
    testAlign<float, 4>();
}

TEST_F(CnmemTest, alignment64) {
    testAlign<double, 8>();
}

TEST_F(CnmemTest, alignment192) {
    testAlign<_24ByteStruct, 24>();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

