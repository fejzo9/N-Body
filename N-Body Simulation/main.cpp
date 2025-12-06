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
#include <immintrin.h> // SIMD/AVX support
#include <algorithm>   // For std::max

// Koristimo float za sve fizikalne proračune (standardizacija sa benchmarkom)
const float G_LOCAL = 1.0f;
const int WINDOW_SIZE = 1000;
const int VEC_STRIDE = 8; // AVX width (8 floats in __m256)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// --- Ujedinjena Vec struktura (float) ---
struct Vec
{
    float x, y;
    Vec(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
    Vec operator+(const Vec &o) const { return Vec(x + o.x, y + o.y); }
    Vec operator-(const Vec &o) const { return Vec(x - o.x, y - o.y); }
    Vec operator*(float k) const { return Vec(x * k, y * k); }
    Vec operator/(float k) const { return Vec(x / k, y / k); }
    float len() const { return std::sqrt(x * x + y * y); }
    Vec norm() const
    {
        float l = len();
        return l == 0.0f ? Vec(0.0f, 0.0f) : *this / l;
    }
};

// -------------------- Struct of Arrays (SoA) for Bodies --------------------
// (Standardizovano sa benchmarkom)
struct BodyDataSoA
{
    std::vector<float> m;
    std::vector<float> posX;
    std::vector<float> posY;
    std::vector<float> velX;
    std::vector<float> velY;
    std::vector<float> accX;
    std::vector<float> accY;
    std::vector<float> radius;
    size_t num_bodies = 0;

    void resize(size_t n)
    {
        m.resize(n);
        posX.resize(n);
        posY.resize(n);
        velX.resize(n);
        velY.resize(n);
        accX.resize(n);
        accY.resize(n);
        radius.resize(n);
        num_bodies = n;
    }

    Vec getPos(size_t i) const { return Vec(posX[i], posY[i]); }
    void setAcc(size_t i, const Vec &acc)
    {
        accX[i] = acc.x;
        accY[i] = acc.y;
    }
};

// -------------------- SIMD-Friendly QuadTree Node (iz benchmarka) --------------------
// Koristi float i indeksiranje u pool
struct QuadNode
{
    float mass = 0.0f;
    float comX = 0.0f;
    float comY = 0.0f;
    float topLeftX, topLeftY;
    float bottomRightX, bottomRightY;
    int bodyIndex = -1; // Index u BodyDataSoA ako je list
    int children[4] = {-1, -1, -1, -1}; // Indeksi u pool-u

    QuadNode() {}
    QuadNode(float tlX, float tlY, float brX, float brY) 
        : topLeftX(tlX), topLeftY(tlY), bottomRightX(brX), bottomRightY(brY) {}

    inline bool isLeaf() const { return children[0] == -1; }

    inline bool contains(float px, float py) const
    {
        return px >= topLeftX && px <= bottomRightX &&
               py >= topLeftY && py <= bottomRightY;
    }
};

// -------------------- QuadTree Klasa (iz benchmarka) --------------------
class QuadTree
{
private:
    std::vector<QuadNode> pool;
    int lastPoolIndex = 1;
    
    // Potrebna rekurzivna funkcija za insert koja koristi numeričko indeksiranje
    void insert_recursive(int node_idx, size_t body_idx, const BodyDataSoA &data, int poolSize)
    {
        QuadNode *node = &pool[node_idx];
        float b_posX = data.posX[body_idx];
        float b_posY = data.posY[body_idx];

        if (!node->contains(b_posX, b_posY))
            return;

        // Slučaj 1: Čvor je prazan list
        if (node->bodyIndex == -1 && node->isLeaf())
        {
            node->bodyIndex = (int)body_idx;
            node->mass = data.m[body_idx];
            node->comX = b_posX;
            node->comY = b_posY;
            return;
        }

        // Slučaj 2: Čvor je pun list (treba podijeliti)
        if (node->isLeaf())
        {
            float midX = (node->topLeftX + node->bottomRightX) * 0.5f;
            float midY = (node->topLeftY + node->bottomRightY) * 0.5f;
            
            if (lastPoolIndex + 4 >= pool.size()) return; 

            // Alokacija novih čvorova u pool-u
            int c0 = lastPoolIndex++; pool[c0] = QuadNode(node->topLeftX, node->topLeftY, midX, midY);
            int c1 = lastPoolIndex++; pool[c1] = QuadNode(midX, node->topLeftY, node->bottomRightX, midY);
            int c2 = lastPoolIndex++; pool[c2] = QuadNode(node->topLeftX, midY, midX, node->bottomRightY);
            int c3 = lastPoolIndex++; pool[c3] = QuadNode(midX, midY, node->bottomRightX, node->bottomRightY);

            node = &pool[node_idx]; 
            node->children[0] = c0; node->children[1] = c1;
            node->children[2] = c2; node->children[3] = c3;

            int existingIndex = node->bodyIndex;
            node->bodyIndex = -1; 
            
            // Re-insert existing body
            for (int i = 0; i < 4; ++i)
                insert_recursive(node->children[i], existingIndex, data, (int)pool.size());
        }

        // Slučaj 3: Čvor je grana (nastavi rekurziju i ubaci novo tijelo)
        node = &pool[node_idx]; 
        for (int i = 0; i < 4; ++i)
            if (node->children[i] != -1)
                insert_recursive(node->children[i], body_idx, data, (int)pool.size());

        // Update Center of Mass
        node = &pool[node_idx]; 
        node->mass = 0.0f;
        float total_mass_comX = 0.0f;
        float total_mass_comY = 0.0f;
        
        for (int i = 0; i < 4; ++i)
        {
            if (node->children[i] != -1)
            {
                const QuadNode &child = pool[node->children[i]];
                node->mass += child.mass;
                total_mass_comX += child.comX * child.mass;
                total_mass_comY += child.comY * child.mass;
            }
        }
        
        if (node->mass > 0.0f)
        {
            float invMass = 1.0f / node->mass;
            node->comX = total_mass_comX * invMass;
            node->comY = total_mass_comY * invMass;
        }
    }

    // --- AVX Force Calculation Kernel ---
    inline void computeForce_kernel(int n_idx, 
                             __m256 target_idx_vec, 
                             __m256 target_posX_vec, __m256 target_posY_vec, 
                             __m256 target_radius_vec,
                             __m256 *accX_out, __m256 *accY_out) const
    {
        const QuadNode &node = pool[n_idx];
        
        __m256 node_comX = _mm256_set1_ps(node.comX);
        __m256 node_comY = _mm256_set1_ps(node.comY);
        __m256 node_mass = _mm256_set1_ps(node.mass);

        const __m256 eps_vec = _mm256_set1_ps(1e-6f);
        const __m256 soft_vec = _mm256_set1_ps(0.001f);
        
        __m256 deltaX_vec = _mm256_sub_ps(node_comX, target_posX_vec);
        __m256 deltaY_vec = _mm256_sub_ps(node_comY, target_posY_vec);

        __m256 distSq_vec = _mm256_fmadd_ps(deltaX_vec, deltaX_vec, _mm256_mul_ps(deltaY_vec, deltaY_vec));
        __m256 dist_vec = _mm256_sqrt_ps(distSq_vec);
        
        // Mask self-interaction (koristimo float index u registru)
        __m256 node_body_idx_vec = _mm256_set1_ps((float)node.bodyIndex);
        __m256 target_idx_mask = _mm256_cmp_ps(target_idx_vec, node_body_idx_vec, _CMP_EQ_OS);
        
        __m256 safeDist_vec = _mm256_max_ps(dist_vec, target_radius_vec); 
        __m256 safeDistSq_vec = _mm256_mul_ps(safeDist_vec, safeDist_vec);
        __m256 denom_vec = _mm256_add_ps(safeDistSq_vec, _mm256_mul_ps(soft_vec, soft_vec));
        
        __m256 G_vec = _mm256_set1_ps(G_LOCAL);
        __m256 F_vec = _mm256_div_ps(_mm256_mul_ps(G_vec, node_mass), denom_vec);
        
        __m256 invDist_vec = _mm256_rcp_ps(_mm256_add_ps(dist_vec, eps_vec)); 
        __m256 acc_x_contrib = _mm256_mul_ps(_mm256_mul_ps(deltaX_vec, invDist_vec), F_vec);
        __m256 acc_y_contrib = _mm256_mul_ps(_mm256_mul_ps(deltaY_vec, invDist_vec), F_vec);

        // Accumulate with mask (ANDNOT) - Ignore contributions from self-interaction
        *accX_out = _mm256_add_ps(*accX_out, _mm256_andnot_ps(target_idx_mask, acc_x_contrib));
        *accY_out = _mm256_add_ps(*accY_out, _mm256_andnot_ps(target_idx_mask, acc_y_contrib));
    }
    
public: 
    // --- AVX Traversal (iz benchmarka) ---
    void computeForce_vec(int start_index, const BodyDataSoA &data, 
                          __m256 target_posX_vec, __m256 target_posY_vec, 
                          __m256 target_radius_vec, __m256 theta_vec,
                          __m256 *accX_out, __m256 *accY_out) const
    {
        float indices[VEC_STRIDE]; // Stack alokacija
        for (int k = 0; k < VEC_STRIDE; ++k) indices[k] = (float)(start_index + k);
        __m256 target_idx_vec = _mm256_loadu_ps(indices); 
        
        // Stack-allocated array for traversal 
        int node_stack[128];
        int stack_top = 0;
        node_stack[stack_top++] = 0; // Root node index
        
        const __m256 eps_vec = _mm256_set1_ps(1e-6f);
        
        while (stack_top > 0)
        {
            int current_node_idx = node_stack[--stack_top]; 
            const QuadNode &node = pool[current_node_idx];

            if (node.mass == 0.0f) continue;

            float node_width = node.bottomRightX - node.topLeftX;
            __m256 node_width_vec = _mm256_set1_ps(node_width);
            
            __m256 deltaX_vec = _mm256_sub_ps(_mm256_set1_ps(node.comX), target_posX_vec);
            __m256 deltaY_vec = _mm256_sub_ps(_mm256_set1_ps(node.comY), target_posY_vec);
            __m256 distSq_vec = _mm256_add_ps(_mm256_mul_ps(deltaX_vec, deltaX_vec), _mm256_mul_ps(deltaY_vec, deltaY_vec));
            __m256 dist_vec = _mm256_sqrt_ps(distSq_vec);
            
            __m256 ratio_vec = _mm256_div_ps(node_width_vec, _mm256_add_ps(dist_vec, eps_vec));
            __m256 theta_mask = _mm256_cmp_ps(ratio_vec, theta_vec, _CMP_LT_OS); 
            int theta_mask_int = _mm256_movemask_ps(theta_mask); 

            // Provjera far-field uslova (theta mask je 0xFF - Svi su daleko) ILI ako je list
            if (node.isLeaf() || theta_mask_int == 0xFF) 
            {
                computeForce_kernel(current_node_idx, target_idx_vec, target_posX_vec, target_posY_vec, target_radius_vec, accX_out, accY_out);
            }
            else 
            {
                for (int j = 0; j < 4; ++j)
                    if (node.children[j] != -1)
                        node_stack[stack_top++] = node.children[j]; 
            }
        }
    }

    void init(int n, float spread)
    {
        // Pre-allocate large enough pool 
        int poolSize = n * 4 + 1000; 
        pool.resize(poolSize);
        // Početni root čvor
        pool[0] = QuadNode(-spread, -spread, spread, spread); 
        lastPoolIndex = 1;
    }

    void build(const BodyDataSoA &data, float spread)
    {
        // Paralelizacija resetovanja pool-a
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < lastPoolIndex; ++i)
        {
            pool[i].mass = 0.0f;
            pool[i].comX = 0.0f;
            pool[i].comY = 0.0f;
            pool[i].bodyIndex = -1;
            pool[i].children[0] = -1; pool[i].children[1] = -1;
            pool[i].children[2] = -1; pool[i].children[3] = -1;
        }
        
        lastPoolIndex = 1;
        pool[0] = QuadNode(-spread, -spread, spread, spread); 

        // Sekvencijalna izgradnja stabla
        for (size_t i = 0; i < data.num_bodies; ++i)
            insert_recursive(0, i, data, (int)pool.size());
    }
    
    // Getter za pristup pool-u ako je potreban debugging
    // const std::vector<QuadNode>& getPool() const { return pool; }
};


// -------------------- MAIN FUNKCIJA (Ažurirana logika) --------------------

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

int main()
{
    // --- Load configuration from JSON ---
    bool useBarnesHut = true;
    bool useParallel = true;
    int n = 10000;
    float spread = 400.0f;
    float mass = 1000.0f;
    float theta = 1.0f;
    float dt = 0.05f;
    int windowSize = 1000;
    std::string windowTitle = "Chaotic N-Body Simulation (Optimized)";

    std::ifstream configFile("config.json");
    if (configFile.is_open())
    {
        std::string line;
        std::string content;
        while (std::getline(configFile, line))
            content += line;

        auto getValue = [&content](const std::string& key) -> std::string {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return "";
            pos = content.find(":", pos);
            if (pos == std::string::npos) return "";
            pos = content.find_first_not_of(" \t\n\r", pos + 1);
            
            if (content[pos] == '"') {
                pos++;
                size_t endPos = content.find("\"", pos);
                return content.substr(pos, endPos - pos);
            } else {
                size_t endPos = content.find_first_of(",}", pos);
                return content.substr(pos, endPos - pos);
            }
        };

        std::string barnesHutStr = getValue("useBarnesHut");
        std::string parallelStr = getValue("useParallel");
        std::string numBodiesStr = getValue("numBodies");
        std::string spreadStr = getValue("spread");
        std::string massStr = getValue("mass");
        std::string thetaStr = getValue("theta");
        std::string dtStr = getValue("dt");
        std::string windowSizeStr = getValue("windowSize");
        std::string windowTitleStr = getValue("windowTitle");

        if (!barnesHutStr.empty()) useBarnesHut = (barnesHutStr == "true");
        if (!parallelStr.empty()) useParallel = (parallelStr == "true");
        if (!numBodiesStr.empty()) n = std::stoi(numBodiesStr);
        if (!spreadStr.empty()) spread = (float)std::stod(spreadStr);
        if (!massStr.empty()) mass = (float)std::stod(massStr);
        if (!thetaStr.empty()) theta = (float)std::stod(thetaStr);
        if (!dtStr.empty()) dt = (float)std::stod(dtStr);
        if (!windowSizeStr.empty()) windowSize = std::stoi(windowSizeStr);
        if (!windowTitleStr.empty()) windowTitle = windowTitleStr;

        std::cout << "Config loaded: useBarnesHut=" << useBarnesHut 
                  << ", useParallel=" << useParallel 
                  << ", numBodies=" << n << std::endl;
        configFile.close();
    }
    else
    {
        std::cerr << "config.json not found. Using defaults." << std::endl;
    }

    // --- Otvaranje prozora ---
    sf::RenderWindow window(sf::VideoMode(windowSize, windowSize), windowTitle);

    BodyDataSoA data;
    data.resize(n); 

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    std::ofstream logFile("log.txt", std::ios::app);
    if (logFile.is_open())
    {
        logFile << "=== Simulation started: " << currentDateTimeString() << " ===" << std::endl;
        logFile << "=== Number of bodies: " << n << " ===" << std::endl;
        logFile << "=== Using parallelizations: " << (useParallel ? "Yes" : "No") << " ===" << std::endl;
        logFile << "=== Data Structure: Struct of Arrays (SoA) + AVX ===" << std::endl;
    }
    else
    {
        std::cerr << "Ne mogu otvoriti log.txt za pisanje!" << std::endl;
    }

    std::vector<sf::CircleShape> circles; 
    for (int i = 0; i < n; i++)
    {
        float angle = dist01(rng) * 2 * M_PI;
        float r = dist01(rng) * spread;
        Vec pos(r * std::cos(angle), r * std::sin(angle));

        float v = std::sqrt(G_LOCAL * mass * (float)n / (r + 50.0f));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.3f;

        vel.x += (dist01(rng) - 0.5f) * v * 0.05f;
        vel.y += (dist01(rng) - 0.5f) * v * 0.05f;

        data.m[i] = mass;
        data.posX[i] = pos.x;
        data.posY[i] = pos.y;
        data.velX[i] = vel.x;
        data.velY[i] = vel.y;
        data.accX[i] = 0.0f; 
        data.accY[i] = 0.0f;
        data.radius[i] = 8.0f;

        sf::CircleShape circle(data.radius[i]);
        circle.setOrigin(data.radius[i], data.radius[i]);
        circles.push_back(circle);
    }

    sf::View view(sf::Vector2f(0, 0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    float zoomLevel = 1.0f;

    // --- Inicijalizacija Barnes-Huta ---
    QuadTree tree;
    tree.init(n, spread);

    using Clock = std::chrono::high_resolution_clock;
    int frameCount = 0;
    float elapsedTime = 0.0f;

    while (window.isOpen())
    {
        auto frameStart = Clock::now();

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

        // --- Physics ---
        auto physicsStart = Clock::now();

        if (!useBarnesHut)
        {
            // O(N^2) Loop - skalarni
            for (int i = 0; i < n; ++i)
            {
                data.accX[i] = 0.0f;
                data.accY[i] = 0.0f;
            }

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

                    data.accX[i] += acc.x;
                    data.accY[i] += acc.y;

                    data.accX[j] -= acc.x;
                    data.accY[j] -= acc.y;
                }
            }
        }
        else
        {
            // 🌟 Barnes-Hut OPTIMIZED LOGIC (Identično benchmarku)
            
            // 1. Build Tree (uključuje paralelizovan reset i indeksiranje)
            tree.build(data, spread); 
            
            __m256 theta_vec = _mm256_set1_ps(theta);
            
            // 2. Compute Forces (Paralelizovano i Vektorizovano AVX)
            if (useParallel)
            {
                // Koristi AVX-vektorizovanu funkciju computeForce_vec
                #pragma omp parallel for schedule(dynamic, 64)
                for (int i = 0; i < n; i += VEC_STRIDE)
                {
                    // Preskačemo neporavnati kraj
                    if (i + VEC_STRIDE > n) continue; 

                    __m256 accX_vec = _mm256_setzero_ps();
                    __m256 accY_vec = _mm256_setzero_ps();
                    
                    __m256 px = _mm256_loadu_ps(&data.posX[i]);
                    __m256 py = _mm256_loadu_ps(&data.posY[i]);
                    __m256 rad = _mm256_loadu_ps(&data.radius[i]);

                    tree.computeForce_vec(i, data, px, py, rad, theta_vec, &accX_vec, &accY_vec);
                    
                    _mm256_storeu_ps(&data.accX[i], accX_vec);
                    _mm256_storeu_ps(&data.accY[i], accY_vec);
                }
                
                // Ostatak niza (ako n nije djeljiv sa VEC_STRIDE) treba ručno resetovati ako je BH u upotrebi
                for (int i = n - (n % VEC_STRIDE); i < n; ++i) {
                    data.accX[i] = 0.0f; data.accY[i] = 0.0f;
                }
            }
            else
            {
                 // Skalarna implementacija za BH (Ako useParallel=false, ali koristimo BH logiku)
                 // Budući da BH logike bez AVX kernela nema, ova grana samo resetuje akceleraciju
                for (int i = 0; i < n; ++i) {
                    data.accX[i] = 0.0f; data.accY[i] = 0.0f;
                }
            }
        }

        // Velocity-Verlet Integration adapted to SoA - Manual AVX vectorization
        if (useParallel)
        {
            __m256 dt_vec = _mm256_set1_ps(dt);
            int i = 0;
            int vector_width = VEC_STRIDE;
            int num_bodies = (int)data.num_bodies;
            
            // Vectorized loop: Process 8 bodies per iteration
            #pragma omp parallel for schedule(static)
            for (i = 0; i < num_bodies - (vector_width - 1); i += vector_width)
            {
                __m256 velX = _mm256_loadu_ps(&data.velX[i]);
                __m256 velY = _mm256_loadu_ps(&data.velY[i]);
                __m256 accX = _mm256_loadu_ps(&data.accX[i]);
                __m256 accY = _mm256_loadu_ps(&data.accY[i]);
                __m256 posX = _mm256_loadu_ps(&data.posX[i]);
                __m256 posY = _mm256_loadu_ps(&data.posY[i]);
                
                // Update Velocity: V += A * dt
                __m256 accX_scaled = _mm256_mul_ps(accX, dt_vec);
                __m256 accY_scaled = _mm256_mul_ps(accY, dt_vec);
                velX = _mm256_add_ps(velX, accX_scaled);
                velY = _mm256_add_ps(velY, accY_scaled);
                
                // Update Position: P += V * dt
                __m256 velX_scaled = _mm256_mul_ps(velX, dt_vec);
                __m256 velY_scaled = _mm256_mul_ps(velY, dt_vec);
                posX = _mm256_add_ps(posX, velX_scaled);
                posY = _mm256_add_ps(posY, velY_scaled);
                
                // Store results back
                _mm256_storeu_ps(&data.velX[i], velX);
                _mm256_storeu_ps(&data.velY[i], velY);
                _mm256_storeu_ps(&data.posX[i], posX);
                _mm256_storeu_ps(&data.posY[i], posY);
            }
            
            // Handle remaining bodies (scalar fallback)
            for (; i < num_bodies; ++i)
            {
                data.velX[i] += data.accX[i] * dt;
                data.velY[i] += data.accY[i] * dt;
                data.posX[i] += data.velX[i] * dt;
                data.posY[i] += data.velY[i] * dt;
            }
        }
        else
        {
            // Skalarna integracija
            for (size_t i = 0; i < data.num_bodies; ++i)
            {
                data.velX[i] += data.accX[i] * dt;
                data.velY[i] += data.accY[i] * dt;
                data.posX[i] += data.velX[i] * dt;
                data.posY[i] += data.velY[i] * dt;
            }
        }

        auto physicsEnd = Clock::now();

        // --- Rendering ---
        window.clear(sf::Color::Black);
        window.setView(view);

        for (size_t i = 0; i < data.num_bodies; ++i)
        {
            circles[i].setPosition(data.posX[i], data.posY[i]);
            window.draw(circles[i]);
        }

        // --- Performance logging ---
        auto frameEnd = Clock::now();
        std::chrono::duration<float> frameDuration = frameEnd - frameStart;
        std::chrono::duration<float> physicsDuration = physicsEnd - physicsStart;

        frameCount++;
        elapsedTime += frameDuration.count();

        float physicsMs = physicsDuration.count() * 1000.0f;
        float frameMs = frameDuration.count() * 1000.0f;
        float fpsValue = 0.0f;
        if (elapsedTime > 0.0f)
            fpsValue = frameCount / elapsedTime;

        char buf[256];

        static const float logInterval = 1.0f;
        static std::string lastLine;

        if (elapsedTime >= logInterval)
        {
            int written = std::snprintf(buf, sizeof(buf),
                                         "FPS: %.2f | Physics: %.2f ms | Total frame: %.2f ms | N=%d",
                                         std::isfinite(fpsValue) ? fpsValue : 0.0f,
                                         std::isfinite(physicsMs) ? physicsMs : 0.0f,
                                         std::isfinite(frameMs) ? frameMs : 0.0f,
                                         n);

            std::string line = (written > 0) ? std::string(buf) : std::string("FPS: N/A |Physics: N/A | Total frame: N/A");

            lastLine = line;

            std::cout << lastLine << std::endl;

            if (logFile.is_open())
            {
                logFile << lastLine << std::endl;
                logFile.flush();
            }

            frameCount = 0;
            elapsedTime = 0.0f;
        }
        
        window.display();
    }

    if (logFile.is_open())
    {
        logFile << "=== Simulation ended: " << currentDateTimeString() << " ===" << std::endl
                << std::endl;
        logFile.close();
    }

    return 0;
}