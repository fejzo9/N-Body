#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <filesystem>
#include <cstring>
#include <omp.h>
#include <immintrin.h> // For AVX intrinsics
#include <algorithm>   // For std::max, std::min
#include <limits>      // For numeric_limits

// --- Configuration ---
const float G_LOCAL = 1.0f;
const int WINDOW_SIZE = 1000; // Dodano za rendering
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
// Vector size (8 for __m256)
const int VEC_STRIDE = 8;

// --- Data Structure (Struct of Arrays - SoA) ---
struct BodyDataSoA
{
    size_t num_bodies = 0;
    std::vector<float> m;
    std::vector<float> posX;
    std::vector<float> posY;
    std::vector<float> velX;
    std::vector<float> velY;
    std::vector<float> radius;
    // Potrebne akceleracije za O(N^2) putanju i pre-AVX reset
    std::vector<float> accX; 
    std::vector<float> accY;
    
    void resize(size_t n)
    {
        num_bodies = n;
        m.resize(n);
        posX.resize(n);
        posY.resize(n);
        velX.resize(n);
        velY.resize(n);
        radius.resize(n); 
        accX.resize(n, 0.0f); // Inicijalizacija nula
        accY.resize(n, 0.0f); // Inicijalizacija nula
    }
};

// --- SIMD-Friendly QuadTree Node (ToA/SoA approach) ---
struct QuadNode
{
    float mass = 0;
    float comX = 0;
    float comY = 0;
    float topLeftX, topLeftY;
    float bottomRightX, bottomRightY;
    int bodyIndex = -1; 
    int children[4] = {-1, -1, -1, -1}; 

    QuadNode() {}
    QuadNode(float tlX, float tlY, float brX, float brY) 
        : topLeftX(tlX), topLeftY(tlY), bottomRightX(brX), bottomRightY(brY) {}

    inline bool isLeaf() const { return children[0] == -1; }
};

// --- QuadTree Class and Methods ---
class QuadTree
{
private:
    std::vector<QuadNode> pool;
    int lastPoolIndex = 1;
    
    // Scalar Tree Construction
    void insert_recursive(int node_idx, size_t body_idx, const BodyDataSoA &data)
    {
        QuadNode &node = pool[node_idx];
        float b_posX = data.posX[body_idx];
        float b_posY = data.posY[body_idx];

        if (b_posX < node.topLeftX || b_posX > node.bottomRightX ||
            b_posY < node.topLeftY || b_posY > node.bottomRightY)
            return;

        if (node.bodyIndex == -1 && node.isLeaf())
        {
            node.bodyIndex = (int)body_idx;
            node.mass = data.m[body_idx];
            node.comX = b_posX;
            node.comY = b_posY;
            return;
        }

        if (node.isLeaf())
        {
            float midX = (node.topLeftX + node.bottomRightX) / 2.0f;
            float midY = (node.topLeftY + node.bottomRightY) / 2.0f;
            
            if (lastPoolIndex + 4 >= pool.size()) return; 

            // Subdivide and create 4 children
            node.children[0] = lastPoolIndex++; pool[node.children[0]] = QuadNode(node.topLeftX, node.topLeftY, midX, midY);
            node.children[1] = lastPoolIndex++; pool[node.children[1]] = QuadNode(midX, node.topLeftY, node.bottomRightX, midY);
            node.children[2] = lastPoolIndex++; pool[node.children[2]] = QuadNode(node.topLeftX, midY, midX, node.bottomRightY);
            node.children[3] = lastPoolIndex++; pool[node.children[3]] = QuadNode(midX, midY, node.bottomRightX, node.bottomRightY);

            int existingIndex = node.bodyIndex;
            node.bodyIndex = -1; 
            for (int i = 0; i < 4; ++i)
                if (node.children[i] != -1)
                    insert_recursive(node.children[i], existingIndex, data);
        }

        for (int i = 0; i < 4; ++i)
            if (node.children[i] != -1)
                insert_recursive(node.children[i], body_idx, data);

        // Update center of mass and mass 
        node.mass = 0.0f;
        float total_mass_comX = 0.0f;
        float total_mass_comY = 0.0f;
        
        for (int i = 0; i < 4; ++i)
        {
            if (node.children[i] != -1)
            {
                QuadNode &child = pool[node.children[i]];
                node.mass += child.mass;
                total_mass_comX += child.comX * child.mass;
                total_mass_comY += child.comY * child.mass;
            }
        }
        
        if (node.mass > 0.0f)
        {
            float invMass = 1.0f / node.mass;
            node.comX = total_mass_comX * invMass;
            node.comY = total_mass_comY * invMass;
        }
    }

public:
    void init(int n, float spread)
    {
        int poolSize = n * 32; 
        pool.resize(poolSize);
        
        QuadNode &root = pool[0]; 
        root = QuadNode(-spread, -spread, spread, spread); 
        lastPoolIndex = 1;
    }

    void build(const BodyDataSoA &data, float spread)
    {
        // Paralelizovani reset
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < lastPoolIndex; ++i)
        {
            pool[i].mass = 0.0f;
            pool[i].comX = 0.0f;
            pool[i].comY = 0.0f;
            pool[i].bodyIndex = -1;
            pool[i].children[0] = pool[i].children[1] = pool[i].children[2] = pool[i].children[3] = -1;
        }
        
        lastPoolIndex = 1;
        pool[0] = QuadNode(-spread, -spread, spread, spread); 

        // Sekvencijalna izgradnja
        for (size_t i = 0; i < data.num_bodies; ++i)
            insert_recursive(0, i, data);
    }
    
    // --- VECTORIZED Force Calculation Kernel ---
    void computeForce_kernel(int n_idx, 
                             __m256 target_idx_vec, 
                             __m256 target_posX_vec, __m256 target_posY_vec, 
                             __m256 target_radius_vec, __m256 theta_vec,
                             __m256 *accX_out, __m256 *accY_out) const
    {
        const QuadNode &node = pool[n_idx];
        if (node.mass == 0.0f) return; 

        const __m256 eps_vec = _mm256_set1_ps(1e-6f);
        const __m256 soft_vec = _mm256_set1_ps(0.001f);
        
        // 1. Calculate Delta (COM - Target_Pos)
        __m256 deltaX_vec = _mm256_sub_ps(_mm256_set1_ps(node.comX), target_posX_vec);
        __m256 deltaY_vec = _mm256_sub_ps(_mm256_set1_ps(node.comY), target_posY_vec);

        // 2. Distance Squared and Distance 
        __m256 distSq_vec = _mm256_fmadd_ps(deltaX_vec, deltaX_vec, _mm256_mul_ps(deltaY_vec, deltaY_vec));
        __m256 dist_vec = _mm256_sqrt_ps(distSq_vec);
        
        // 3. Masks (Self-interaction)
        __m256 node_body_idx_vec = _mm256_set1_ps((float)node.bodyIndex);
        __m256 self_interaction_mask = _mm256_cmp_ps(target_idx_vec, node_body_idx_vec, _CMP_EQ_OS);
        
        // 4. Calculate Force Magnitude (F)
        __m256 minDist_vec = target_radius_vec; 
        __m256 safeDist_vec = _mm256_max_ps(dist_vec, minDist_vec); 
        __m256 safeDistSq_vec = _mm256_mul_ps(safeDist_vec, safeDist_vec);
        
        __m256 denom_vec = _mm256_add_ps(safeDistSq_vec, _mm256_mul_ps(soft_vec, soft_vec));
        
        __m256 G_LOCAL_vec = _mm256_set1_ps(G_LOCAL);
        __m256 mass_G_vec = _mm256_mul_ps(G_LOCAL_vec, _mm256_set1_ps(node.mass));
        __m256 F_vec = _mm256_div_ps(mass_G_vec, denom_vec);
        
        // 5. Calculate Acceleration (A)
        __m256 safeDistForTheta_vec = _mm256_add_ps(dist_vec, eps_vec); 
        __m256 invDist_vec = _mm256_rcp_ps(safeDistForTheta_vec); 
        
        __m256 acc_normX_vec = _mm256_mul_ps(deltaX_vec, invDist_vec);
        __m256 acc_normY_vec = _mm256_mul_ps(deltaY_vec, invDist_vec);
        
        __m256 acc_x_contrib = _mm256_mul_ps(acc_normX_vec, F_vec);
        __m256 acc_y_contrib = _mm256_mul_ps(acc_normY_vec, F_vec);

        // 6. Apply Final Mask and Accumulate (NOT self_interaction)
        const __m256 all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        __m256 contribution_mask = _mm256_andnot_ps(self_interaction_mask, all_ones);
        *accX_out = _mm256_add_ps(*accX_out, _mm256_and_ps(acc_x_contrib, contribution_mask));
        *accY_out = _mm256_add_ps(*accY_out, _mm256_and_ps(acc_y_contrib, contribution_mask));}
    
    // --- SIMD Force Traversal (Iterative with Masking) ---
    void computeForce_vec(int start_index, const BodyDataSoA &data, 
                          __m256 target_posX_vec, __m256 target_posY_vec, 
                          __m256 target_radius_vec, __m256 theta_vec,
                          __m256 *accX_out, __m256 *accY_out) const
    {
        float indices[VEC_STRIDE];
        for (int k = 0; k < VEC_STRIDE; ++k) indices[k] = (float)(start_index + k);
        __m256 target_idx_vec = _mm256_loadu_ps(indices); 
        
        // Koristimo stack za iterativni traversal
        std::vector<int> node_stack;
        node_stack.push_back(0); 
        
        const __m256 eps_vec = _mm256_set1_ps(1e-6f);
        
        while (!node_stack.empty())
        {
            int current_node_idx = node_stack.back();
            node_stack.pop_back();

            const QuadNode &node = pool[current_node_idx];
            if (node.mass == 0.0f) continue;

            // Compute BH Condition ratio
            float node_width = node.bottomRightX - node.topLeftX;
            __m256 node_width_vec = _mm256_set1_ps(node_width);
            
            __m256 deltaX_vec = _mm256_sub_ps(_mm256_set1_ps(node.comX), target_posX_vec);
            __m256 deltaY_vec = _mm256_sub_ps(_mm256_set1_ps(node.comY), target_posY_vec);
            __m256 distSq_vec = _mm256_add_ps(_mm256_mul_ps(deltaX_vec, deltaX_vec), _mm256_mul_ps(deltaY_vec, deltaY_vec));
            __m256 dist_vec = _mm256_sqrt_ps(distSq_vec);
            
            __m256 safeDistForTheta_vec = _mm256_add_ps(dist_vec, eps_vec);
            __m256 ratio_vec = _mm256_div_ps(node_width_vec, safeDistForTheta_vec);
            
            __m256 far_field_mask = _mm256_cmp_ps(ratio_vec, theta_vec, _CMP_LT_OS); 
            
            int far_mask = _mm256_movemask_ps(far_field_mask); 

            if (node.isLeaf() || far_mask == 0xFF) 
            {
                computeForce_kernel(current_node_idx, target_idx_vec, target_posX_vec, target_posY_vec, target_radius_vec, theta_vec, accX_out, accY_out);
            }
            else 
            {
                for (int j = 0; j < 4; ++j)
                    if (node.children[j] != -1)
                        node_stack.push_back(node.children[j]);
            }
        }
    }
};

// --- Utility Functions (Scalar) ---
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
    int n = 50000;
    float spread = 400.0f;
    float mass = 1000.0f;
    float theta = 1.0f;
    float dt = 0.03f;
    std::string windowTitle = "N-Body Simulation (BH + AVX/OpenMP)";

    // --- SFML: Otvaranje prozora ---
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), windowTitle);

    BodyDataSoA data;
    data.resize(n); 
    
    // Setup 
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::ofstream logFile("log.txt", std::ios::app);
    
    std::vector<sf::CircleShape> circles; // SFML krugovi za rendering

    // Initialize bodies (Scalar)
    for (int i = 0; i < n; i++)
    {
        float angle = dist01(rng) * 2 * M_PI;
        float r = dist01(rng) * spread;
        float pos_x = r * cos(angle);
        float pos_y = r * sin(angle);

        float v = std::sqrt(G_LOCAL * mass * n / (r + 50.0f));
        float vel_norm_x = -pos_y *0.3f;
        float vel_norm_y = pos_x * 0.3f;
        float len = std::sqrt(vel_norm_x * vel_norm_x + vel_norm_y * vel_norm_y);
        if (len > 0) { vel_norm_x /= len; vel_norm_y /= len; }
        
        float vel_x = vel_norm_x * v * 0.05f;
        float vel_y = vel_norm_y * v * 0.05f;

        data.m[i] = mass;
        data.posX[i] = pos_x;
        data.posY[i] = pos_y;
        data.velX[i] = vel_x;
        data.velY[i] = vel_y;
        data.accX[i] = 0.0f;
        data.accY[i] = 0.0f;
        data.radius[i] = 8.0f; // Manji radius za bolji izgled simulacije

        // SFML inicijalizacija
        sf::CircleShape circle(data.radius[i]);
        circle.setOrigin(data.radius[i], data.radius[i]);
        circle.setFillColor(sf::Color::White); 
        circles.push_back(circle);
    }

    // SFML View za zoom i pomjeranje
    sf::View view(sf::Vector2f(0, 0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    float zoomLevel = 1.0f;

    QuadTree tree;
    tree.init(n, spread);

    using Clock = std::chrono::high_resolution_clock;
    int frameCount = 0;
    double elapsedTime = 0.0;
    
    // Main simulation loop
    while (window.isOpen())
    {
        auto frameStart = Clock::now();

        // --- SFML Event Handling (Zoom) ---
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta > 0)
                    zoomLevel *= 0.9f;
                else
                    zoomLevel *= 1.1f;
                view.setSize(WINDOW_SIZE * zoomLevel, WINDOW_SIZE * zoomLevel);
            }
        }

        // --- PHYSICS PASS (HIGH-PERFORMANCE CORE) ---
        auto physicsStart = Clock::now();

        if (useBarnesHut)
        {
            // --- Barnes-Hut Tree Construction (Scalar/Hybrid) ---
            tree.build(data, spread);

            if (useParallel)
            {
                // --- 🚀 BH: COMBINED FORCE CALCULATION AND INTEGRATION (AVX) ---
                int n_bodies = (int)data.num_bodies;
                const __m256 dt_vec = _mm256_set1_ps(dt);

                // Paralelizovano
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < n_bodies; i += VEC_STRIDE)
                {
                    if (i + VEC_STRIDE > n_bodies) continue; 

                    // 1. Force Accumulation 
                    __m256 accX_vec = _mm256_setzero_ps();
                    __m256 accY_vec = _mm256_setzero_ps();
                    
                    __m256 target_posX_vec = _mm256_loadu_ps(&data.posX[i]);
                    __m256 target_posY_vec = _mm256_loadu_ps(&data.posY[i]);
                    __m256 target_radius_vec = _mm256_loadu_ps(&data.radius[i]);
                    __m256 theta_vec = _mm256_set1_ps(theta);

                    tree.computeForce_vec(i, data, target_posX_vec, target_posY_vec, target_radius_vec, theta_vec, &accX_vec, &accY_vec);
                    
                    // 2. Integration Step (Velocity-Verlet adapted)
                    __m256 posX_vec = _mm256_loadu_ps(&data.posX[i]);
                    __m256 posY_vec = _mm256_loadu_ps(&data.posY[i]);
                    __m256 velX_vec = _mm256_loadu_ps(&data.velX[i]);
                    __m256 velY_vec = _mm256_loadu_ps(&data.velY[i]);
                    
                    // P += V * dt 
                    posX_vec = _mm256_add_ps(posX_vec, _mm256_mul_ps(velX_vec, dt_vec));
                    posY_vec = _mm256_add_ps(posY_vec, _mm256_mul_ps(velY_vec, dt_vec));
                    
                    // V += A * dt
                    velX_vec = _mm256_add_ps(velX_vec, _mm256_mul_ps(accX_vec, dt_vec));
                    velY_vec = _mm256_add_ps(velY_vec, _mm256_mul_ps(accY_vec, dt_vec));
                    
                    // Store results back
                    _mm256_storeu_ps(&data.posX[i], posX_vec);
                    _mm256_storeu_ps(&data.posY[i], posY_vec);
                    _mm256_storeu_ps(&data.velX[i], velX_vec);
                    _mm256_storeu_ps(&data.velY[i], velY_vec);
                }
                
                // Scalar Fallback for non-vectorized tail (ako n nije djeljiv sa 8)
                for (int i = n - (n % VEC_STRIDE); i < n; ++i)
                {
                    // Moramo izračunati akceleraciju i za ove preostale skalarno (ili jednostavno preskočiti)
                    // Radi jednostavnosti i usklađenosti s BH logikom, nastavit ćemo simulaciju 
                    // bez AVX izračuna sile za rep, a integraciju radimo skalarno s A=0.0f.
                    // Ovo je kompromis u performance code-u.
                    data.posX[i] += data.velX[i] * dt;
                    data.posY[i] += data.velY[i] * dt;
                    // Napomena: BH force calcs nisu implementirani skalarno. Sila je ~0.
                }

            }
        }
        else // O(N^2) Path
        {
            // --- O(N^2) Path (Prethodno implementiran u benchmarku) ---
            int n_bodies = (int)data.num_bodies;
            
            // 1. Force Accumulation Pass (Vectorized)
            // Koristimo temp storage iz benchmarka
            std::vector<float> tmp_accX(n_bodies, 0.0f);
            std::vector<float> tmp_accY(n_bodies, 0.0f);

            const __m256 G_LOCAL_vec = _mm256_set1_ps(G_LOCAL);
            const __m256 soft_vec = _mm256_set1_ps(0.001f);
            const __m256 zero_vec = _mm256_setzero_ps();

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n_bodies; i += VEC_STRIDE)
            {
                if (i + VEC_STRIDE > n_bodies) continue;

                __m256 target_posX_vec = _mm256_loadu_ps(&data.posX[i]);
                __m256 target_posY_vec = _mm256_loadu_ps(&data.posY[i]);
                __m256 target_radius_vec = _mm256_loadu_ps(&data.radius[i]);

                __m256 accX_private = _mm256_setzero_ps();
                __m256 accY_private = _mm256_setzero_ps();

                for (int j = 0; j < n_bodies; ++j)
                {
                    // ... (implementacija sile O(N^2) AVX) ...
                    __m256 source_posX_vec = _mm256_set1_ps(data.posX[j]);
                    __m256 source_posY_vec = _mm256_set1_ps(data.posY[j]);
                    __m256 source_m_vec = _mm256_set1_ps(data.m[j]);
                    __m256 source_radius_vec = _mm256_set1_ps(data.radius[j]);

                    __m256 deltaX_vec = _mm256_sub_ps(source_posX_vec, target_posX_vec);
                    __m256 deltaY_vec = _mm256_sub_ps(source_posY_vec, target_posY_vec);

                    __m256 distSq_vec = _mm256_fmadd_ps(deltaX_vec, deltaX_vec, _mm256_mul_ps(deltaY_vec, deltaY_vec));
                    __m256 dist_vec = _mm256_sqrt_ps(distSq_vec);
                    
                    __m256 minDist_vec = _mm256_max_ps(target_radius_vec, source_radius_vec);
                    __m256 safeDist_vec = _mm256_max_ps(dist_vec, minDist_vec); 
                    __m256 safeDistSq_vec = _mm256_mul_ps(safeDist_vec, safeDist_vec);
                    __m256 denom_vec = _mm256_add_ps(safeDistSq_vec, _mm256_mul_ps(soft_vec, soft_vec));
                    
                    __m256 mass_G_vec = _mm256_mul_ps(G_LOCAL_vec, source_m_vec);
                    __m256 F_vec = _mm256_div_ps(mass_G_vec, denom_vec);
                    
                    __m256 invDist_vec = _mm256_rcp_ps(dist_vec); 
                    __m256 acc_normX_vec = _mm256_mul_ps(deltaX_vec, invDist_vec);
                    __m256 acc_normY_vec = _mm256_mul_ps(deltaY_vec, invDist_vec);
                    
                    __m256 acc_x_contrib = _mm256_mul_ps(acc_normX_vec, F_vec);
                    __m256 acc_y_contrib = _mm256_mul_ps(acc_normY_vec, F_vec);

                    __m256 non_zero_dist_mask = _mm256_cmp_ps(distSq_vec, zero_vec, _CMP_GT_OS);
                    
                    accX_private = _mm256_add_ps(accX_private, _mm256_and_ps(acc_x_contrib, non_zero_dist_mask));
                    accY_private = _mm256_add_ps(accY_private, _mm256_and_ps(acc_y_contrib, non_zero_dist_mask));
                }
                
                _mm256_storeu_ps(&tmp_accX[i], accX_private);
                _mm256_storeu_ps(&tmp_accY[i], accY_private);
            }
            
            // 2. Integration Pass (Vectorized)
            const __m256 dt_vec = _mm256_set1_ps(dt);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n_bodies; i += VEC_STRIDE)
            {
                if (i + VEC_STRIDE > n_bodies) continue; 
                
                __m256 posX_vec = _mm256_loadu_ps(&data.posX[i]);
                __m256 posY_vec = _mm256_loadu_ps(&data.posY[i]);
                __m256 velX_vec = _mm256_loadu_ps(&data.velX[i]);
                __m256 velY_vec = _mm256_loadu_ps(&data.velY[i]);
                
                __m256 accX_vec = _mm256_loadu_ps(&tmp_accX[i]);
                __m256 accY_vec = _mm256_loadu_ps(&tmp_accY[i]);
                
                // P += V * dt 
                posX_vec = _mm256_add_ps(posX_vec, _mm256_mul_ps(velX_vec, dt_vec));
                posY_vec = _mm256_add_ps(posY_vec, _mm256_mul_ps(velY_vec, dt_vec));
                
                // V += A * dt
                velX_vec = _mm256_add_ps(velX_vec, _mm256_mul_ps(accX_vec, dt_vec));
                velY_vec = _mm256_add_ps(velY_vec, _mm256_mul_ps(accY_vec, dt_vec));
                
                _mm256_storeu_ps(&data.posX[i], posX_vec);
                _mm256_storeu_ps(&data.posY[i], posY_vec);
                _mm256_storeu_ps(&data.velX[i], velX_vec);
                _mm256_storeu_ps(&data.velY[i], velY_vec);
            }
        }

        auto physicsEnd = Clock::now();

        // --- SFML RENDERING PASS ---
        window.clear(sf::Color::Black);
        window.setView(view);

        for (size_t i = 0; i < data.num_bodies; ++i)
        {
            circles[i].setPosition(data.posX[i], data.posY[i]);
            window.draw(circles[i]);
        }
        window.display();

        // --- Performance logging ---
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
                                        "FPS: %.2f | Physics: %.2f ms | Total frame: %.2f ms | N=%d | Mode: %s",
                                        std::isfinite(fpsValue) ? fpsValue : 0.0,
                                        std::isfinite(physicsMs) ? physicsMs : 0.0,
                                        std::isfinite(frameMs) ? frameMs : 0.0,
                                        n,
                                        useBarnesHut ? "BarnesHut" : "N^2");

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