#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstring>
#include <omp.h>
#include <immintrin.h> // For AVX intrinsics

// Use float for G to match BodyDataSoA
const float G_LOCAL = 1.0f; 

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// --- Helper for Horizontal Sum (Corrected AVX Intrinsics) ---
// This function is still included for the O(N^2) fallback path, if needed.
float hsum_ps_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 temp1 = _mm_hadd_ps(v128, v128); 
    __m128 temp2 = _mm_hadd_ps(temp1, temp1);
    return _mm_cvtss_f32(temp2);
}
// --- End Helper ---

// --- Minimal Vec Struct ---
struct Vec
{
    float x, y;
    Vec(float x = 0, float y = 0) : x(x), y(y) {}
    Vec operator+(const Vec &o) const { return Vec(x + o.x, y + o.y); }
    Vec operator-(const Vec &o) const { return Vec(x - o.x, y - o.y); }
    Vec operator*(float k) const { return Vec(x * k, y * k); }
    Vec operator/(float k) const { return Vec(x / k, y / k); }
    Vec &operator+=(const Vec &o)
    {
        x += o.x;
        y += o.y;
        return *this;
    }
    float len() const { return std::sqrt(x * x + y * y); }
    Vec norm() const
    {
        float l = len();
        return l == 0 ? Vec(0, 0) : *this / l;
    }
};

// --- Data Structure (Struct of Arrays - SoA) ---
struct BodyDataSoA
{
    // ALL arrays are now unpadded (size N)
    std::vector<float> m;
    std::vector<float> posX;
    std::vector<float> posY;
    std::vector<float> velX;
    std::vector<float> velY;
    std::vector<float> radius;
    
    // accX and accY are REMOVED as they are not needed for Euler/Simplified V-V
    
    size_t num_bodies = 0;

    void resize(size_t n)
    {
        // Resize all vectors to N
        m.resize(n);
        posX.resize(n);
        posY.resize(n);
        velX.resize(n);
        velY.resize(n);
        radius.resize(n); 
        
        // No accX/accY resize needed
        
        num_bodies = n;
    }

    Vec getPos(size_t i) const { return Vec(posX[i], posY[i]); }
};

struct BodyReference
{
    size_t index;
};

// --- QuadTree Structure (Scalar) ---
struct QuadNode
{
    float mass = 0;
    Vec centerOfMass;
    Vec topLeft, bottomRight;
    BodyReference *bodyRef = nullptr;

    QuadNode *children[4] = {nullptr, nullptr, nullptr, nullptr};

    QuadNode() {}
    QuadNode(const Vec &tl, const Vec &br) : topLeft(tl), bottomRight(br) {}

    bool contains(const Vec &p) const
    {
        return p.x >= topLeft.x && p.x <= bottomRight.x &&
               p.y >= topLeft.y && p.y <= bottomRight.y;
    }

    bool isLeaf() const { return children[0] == nullptr; }

    void subdivide(QuadNode *pool, int &poolIndex, int poolSize)
    {
        if (poolIndex + 4 >= poolSize)
            return;
        Vec mid((topLeft.x + bottomRight.x) / 2, (topLeft.y + bottomRight.y) / 2);
        children[0] = &pool[poolIndex++];
        children[0]->topLeft = topLeft;
        children[0]->bottomRight = mid;
        children[1] = &pool[poolIndex++];
        children[1]->topLeft = Vec(mid.x, topLeft.y);
        children[1]->bottomRight = Vec(bottomRight.x, mid.y);
        children[2] = &pool[poolIndex++];
        children[2]->topLeft = Vec(topLeft.x, mid.y);
        children[2]->bottomRight = Vec(mid.x, bottomRight.y);
        children[3] = &pool[poolIndex++];
        children[3]->topLeft = mid;
        children[3]->bottomRight = bottomRight;
    }

    void insert(size_t bodyIndex, const BodyDataSoA &data, BodyReference *bodyRefs, QuadNode *pool, int &poolIndex, int poolSize)
    {
        Vec b_pos = data.getPos(bodyIndex);

        if (!contains(b_pos))
            return;

        if (!bodyRef && isLeaf())
        {
            bodyRef = &bodyRefs[bodyIndex];
            mass = data.m[bodyIndex];
            centerOfMass = b_pos;
            return;
        }

        if (isLeaf())
        {
            subdivide(pool, poolIndex, poolSize);
            if (bodyRef)
            {
                size_t existingIndex = bodyRef->index;
                bodyRef = nullptr;
                for (int i = 0; i < 4; ++i)
                    if (children[i])
                        children[i]->insert(existingIndex, data, bodyRefs, pool, poolIndex, poolSize);
            }
        }

        for (int i = 0; i < 4; ++i)
            if (children[i])
                children[i]->insert(bodyIndex, data, bodyRefs, pool, poolIndex, poolSize);

        mass = 0;
        centerOfMass = Vec(0, 0);
        for (int i = 0; i < 4; ++i)
        {
            if (children[i])
            {
                mass += children[i]->mass;
                centerOfMass += children[i]->centerOfMass * children[i]->mass;
            }
        }
        if (mass > 0)
            centerOfMass = centerOfMass / mass;
    }

    // --- BH Force Calculation ---
    inline void computeForce(size_t b_index, const BodyDataSoA &data, float target_posX, float target_posY, float target_radius, float theta,
                             float &accX_out, float &accY_out)
    {
        if (bodyRef && bodyRef->index == b_index || mass == 0)
            return;

        Vec b_pos(target_posX, target_posY); 
        float b_radius = target_radius;

        Vec delta = centerOfMass - b_pos;
        float dist = delta.len();
        const float eps = 1e-6f;
        float safeDistForTheta = dist + eps;
        float width = bottomRight.x - topLeft.x;

        if (isLeaf() || (width / safeDistForTheta) < theta)
        {
            float minDist = b_radius;
            float safeDist = std::max(dist, minDist);
            float soft = 0.001f;
            
            float F = (G_LOCAL * mass) / (safeDist * safeDist + soft * soft);
            
            if (dist > eps) {
                Vec acc_norm = delta.norm();
                float acc_x_contrib = acc_norm.x * F;
                float acc_y_contrib = acc_norm.y * F;
                
                // Accumulate into private variables
                accX_out += acc_x_contrib;
                accY_out += acc_y_contrib;
            }
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                if (children[i])
                    children[i]->computeForce(b_index, data, target_posX, target_posY, target_radius, theta, accX_out, accY_out);
        }
    }
};

// --- Utility Functions (Omitted for brevity) ---
std::string currentDateTimeString()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&local_tm, &t);
#else
    localtime_r(&t, &local_tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// --- Main Simulation Loop ---
int main()
{
    // --- Configuration ---
    bool useBarnesHut = true;
    bool useParallel = true;
    int n = 10000;
    float spread = 400.0f;
    float mass = 1000.0f;
    float theta = 1.0f;
    float dt = 0.05f;

    // --- Config Loading Logic (omitted) ---

    BodyDataSoA data;
    data.resize(n); 
    
    // NOTE: PADDING_STRIDE is removed

    std::vector<BodyReference> bodyRefs(n);
    for (int i = 0; i < n; ++i)
    {
        bodyRefs[i].index = i;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    std::ofstream logFile("log.txt", std::ios::app);

    // Initialize bodies 
    for (int i = 0; i < n; i++)
    {
        float angle = dist01(rng) * 2 * M_PI;
        float r = dist01(rng) * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        float v = std::sqrt(G_LOCAL * mass * n / (r + 50.0f));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.3f;

        vel.x += (dist01(rng) - 0.5f) * v * 0.05f;
        vel.y += (dist01(rng) - 0.5f) * v * 0.05f;

        data.m[i] = mass;
        data.posX[i] = pos.x;
        data.posY[i] = pos.y;
        data.velX[i] = vel.x;
        data.velY[i] = vel.y;
        
        // No initialization of accX/accY needed
        
        data.radius[i] = 8.0f;
    }


    int poolSize = n * 16;  
    std::vector<QuadNode> pool(poolSize);

    using Clock = std::chrono::high_resolution_clock;
    int frameCount = 0;
    double elapsedTime = 0.0;
    
    // Main simulation loop
    while (true)
    {
        auto frameStart = Clock::now();
        auto physicsStart = Clock::now();

        if (!useBarnesHut)
        {
            // --- O(N^2) Pairwise Force Calculation (This path is now missing accX/accY storage) ---
            // If you use this path, you must re-add local accX/accY calculation and integration
            // inside the inner loop or accept it will not work correctly without the arrays.
            
            // For N-squared, we'll implement a temporary acc array inside this block
            // to show how it should work without global storage.
            std::vector<float> tmp_accX(n, 0.0f);
            std::vector<float> tmp_accY(n, 0.0f);

            // Force calculation
            #pragma omp parallel for schedule(dynamic) 
            for (int i = 0; i < n; ++i)
            {
                for (int j = i + 1; j < n; ++j)
                {
                    Vec delta(data.posX[j] - data.posX[i], data.posY[j] - data.posY[i]);
                    float dist = delta.len();
                    float minDist = data.radius[i] + data.radius[j];
                    float safeDist = std::max(dist, minDist);
                    float F = (G_LOCAL * data.m[j]) / (safeDist * safeDist);
                    Vec acc = delta.norm() * F;
                    
                    tmp_accX[i] += acc.x;
                    tmp_accY[i] += acc.y;
                    
                    tmp_accX[j] -= acc.x;
                    tmp_accY[j] -= acc.y;
                }
            }
            
            // SCALAR Integration Step
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i)
            {
                data.posX[i] += data.velX[i] * dt;
                data.posY[i] += data.velY[i] * dt;
                data.velX[i] += tmp_accX[i] * dt;
                data.velY[i] += tmp_accY[i] * dt;
            }
        }
        else // useBarnesHut is true
        {
            // --- Barnes-Hut Tree Construction (Scalar) ---
            
            static int lastPoolIndex = 1;
            
            // Tree reset 
            for (int i = 0; i < lastPoolIndex; ++i)
            {
                pool[i].mass = 0;
                pool[i].centerOfMass = Vec(0, 0);
                pool[i].bodyRef = nullptr;
                for (int j = 0; j < 4; ++j)
                    pool[i].children[j] = nullptr;
            }
            int poolIndex = 1;

            Vec topLeft(-spread, -spread);
            Vec bottomRight(spread, spread);
            
            QuadNode &root = pool[0]; 
            root.topLeft = topLeft;
            root.bottomRight = bottomRight;

            for (size_t i = 0; i < data.num_bodies; ++i)
                root.insert(i, data, bodyRefs.data(), pool.data(), poolIndex, poolSize);
            
            lastPoolIndex = poolIndex;

            if (useParallel)
            {
                // --- 🚀 FINAL OPTIMIZATION: COMBINED BH FORCE CALCULATION AND INTEGRATION ---
                // Eliminates L3 cache contention and maximizes temporal locality.
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < (int)data.num_bodies; ++i)
                {
                    // 1. Force Accumulation (Thread-private)
                    float accX_private = 0.0f;
                    float accY_private = 0.0f;
                    
                    float target_posX = data.posX[i];
                    float target_posY = data.posY[i];
                    float target_radius = data.radius[i];
                    
                    // Compute force 
                    root.computeForce(i, data, target_posX, target_posY, target_radius, theta, accX_private, accY_private);
                    
                    // 2. Integration Step (Uses private A immediately)
                    // P += V * dt 
                    data.posX[i] += data.velX[i] * dt;
                    data.posY[i] += data.velY[i] * dt;
                    
                    // V += A_new * dt
                    data.velX[i] += accX_private * dt;
                    data.velY[i] += accY_private * dt;
                    
                    // 3. REMOVED: No more write-back to accX/accY arrays.
                }
            }
            else
            {
                 // Scalar BH implementation omitted
            }
        }

        auto physicsEnd = Clock::now();

        // --- Performance logging (Omitted for brevity) ---
        auto frameEnd = Clock::now();
        std::chrono::duration<double> frameDuration = frameEnd - frameStart;
        std::chrono::duration<double> physicsDuration = physicsEnd - physicsStart;

        frameCount++;
        elapsedTime += frameDuration.count();

        double physicsMs = physicsDuration.count() * 1000.0;
        double frameMs = frameDuration.count() * 1000.0;
        double fpsValue = 0.0;
        if (elapsedTime > 0.0)
            fpsValue = frameCount / elapsedTime;

        char buf[256];

        static const double logInterval = 1.0;
        static std::string lastLine;

        if (elapsedTime >= logInterval)
        {
            int written = std::snprintf(buf, sizeof(buf),
                                        "FPS: %.2f | Physics: %.2f ms | Total frame: %.2f ms",
                                        std::isfinite(fpsValue) ? fpsValue : 0.0,
                                        std::isfinite(physicsMs) ? physicsMs : 0.0,
                                        std::isfinite(frameMs) ? frameMs : 0.0);

            std::string line = (written > 0) ? std::string(buf) : std::string("FPS: N/A | Physics: N/A | Total frame: N/A");

            lastLine = line;

            std::cout << lastLine << std::endl;

            if (logFile.is_open())
            {
                logFile << lastLine << std::endl;
                logFile.flush();
            }

            frameCount = 0;
            elapsedTime = 0.0;
        }
    }

    if (logFile.is_open())
    {
        logFile << "=== Benchmark ended: " << currentDateTimeString() << " ===" << std::endl
                << std::endl;
        logFile.close();
    }

    return 0;
}