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

const double G = 6.67430e-3;
const int WINDOW_SIZE = 1000;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vec
{
    double x, y;
    Vec(double x = 0, double y = 0) : x(x), y(y) {}
    Vec operator+(const Vec &o) const { return Vec(x + o.x, y + o.y); }
    Vec operator-(const Vec &o) const { return Vec(x - o.x, y - o.y); }
    Vec operator*(double k) const { return Vec(x * k, y * k); }
    Vec operator/(double k) const { return Vec(x / k, y / k); }
    Vec &operator+=(const Vec &o)
    {
        x += o.x;
        y += o.y;
        return *this;
    }
    Vec &operator-=(const Vec &o)
    {
        x -= o.x;
        y -= o.y;
        return *this;
    }
    double len() const { return std::sqrt(x * x + y * y); }
    Vec norm() const
    {
        double l = len();
        return l == 0 ? Vec(0, 0) : *this / l;
    }
};

// -------------------- Struct of Arrays (SoA) for Bodies --------------------
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
    Vec getVel(size_t i) const { return Vec(velX[i], velY[i]); }
    Vec getAcc(size_t i) const { return Vec(accX[i], accY[i]); }

    void setAcc(size_t i, const Vec &acc)
    {
        accX[i] = acc.x;
        accY[i] = acc.y;
    }
};

struct BodyReference
{
    size_t index;
};

// -------------------- Barnes-Hut Quadtree (Pool) --------------------
struct QuadNode
{
    double mass = 0;
    Vec centerOfMass;
    Vec topLeft, bottomRight;
    BodyReference *bodyRef = nullptr;

    QuadNode *children[4] = {nullptr, nullptr, nullptr, nullptr}; // Pointers for better prefetching

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
        // zaštita od overflow-a pool-a
        if (poolIndex + 4 >= poolSize)
            return;
        Vec mid((topLeft.x + bottomRight.x) / 2, (topLeft.y + bottomRight.y) / 2);
        children[0] = &pool[poolIndex++];
        children[0]->topLeft = topLeft;
        children[0]->bottomRight = mid; // TL
        children[1] = &pool[poolIndex++];
        children[1]->topLeft = Vec(mid.x, topLeft.y);
        children[1]->bottomRight = Vec(bottomRight.x, mid.y); // TR
        children[2] = &pool[poolIndex++];
        children[2]->topLeft = Vec(topLeft.x, mid.y);
        children[2]->bottomRight = Vec(mid.x, bottomRight.y); // BL
        children[3] = &pool[poolIndex++];
        children[3]->topLeft = mid;
        children[3]->bottomRight = bottomRight; // BR
    }

    // Updated insert function to use SoA data
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
                // Re-insert existing body
                size_t existingIndex = bodyRef->index;
                bodyRef = nullptr; // Clear this node's reference
                for (int i = 0; i < 4; ++i)
                    if (children[i])
                        children[i]->insert(existingIndex, data, bodyRefs, pool, poolIndex, poolSize);
            }
        }

        for (int i = 0; i < 4; ++i)
            if (children[i])
                children[i]->insert(bodyIndex, data, bodyRefs, pool, poolIndex, poolSize);

        // Update center of mass
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

    // Updated computeForce function to use SoA data
    void computeForce(size_t b_index, const BodyDataSoA &data, double theta, Vec &acc_out, double G_local)
    {
        if (bodyRef && bodyRef->index == b_index || mass == 0)
            return;

        Vec b_pos = data.getPos(b_index);
        double b_radius = data.radius[b_index];

        Vec delta = centerOfMass - b_pos;
        double dist = delta.len();
        const double eps = 1e-6; // zaštita za dijeljenje i softening
        double safeDistForTheta = dist + eps;
        double width = bottomRight.x - topLeft.x;

        if (isLeaf() || (width / safeDistForTheta) < theta)
        {
            double minDist = b_radius;
            double safeDist = std::max(dist, minDist);
            // softening: dodamo malu vrijednost u nazivnik da spriječimo ekstremne akceleracije
            double soft = 0.001;
            double F = (G_local * mass) / (safeDist * safeDist + soft * soft);
            if (dist > eps)
                acc_out += delta.norm() * F;
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                if (children[i])
                    children[i]->computeForce(b_index, data, theta, acc_out, G_local);
        }
    }
};

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
    // --- Otvaranje prozora ---
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Chaotic N-Body Simulation");
    window.setFramerateLimit(60);

    bool useBarnesHut = true;
    bool useParallel = true;

    // 🌟 USE SoA DATA STRUCTURE
    BodyDataSoA data;
    double G_local = 1.0;
    int n = 10000;
    double spread = 400.0;
    double mass = 1000.0;
    data.resize(n); // Resize all vectors immediately

    // Vector of references needed for QuadTree
    std::vector<BodyReference> bodyRefs(n);
    for (int i = 0; i < n; ++i)
    {
        bodyRefs[i].index = i;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    // --- LOG fajl (append) ---
    std::ofstream logFile("log.txt", std::ios::app);
    if (logFile.is_open())
    {
        logFile << "=== Simulation started: " << currentDateTimeString() << " ===" << std::endl;
        logFile << "=== Number of bodies: " << n << " ===" << std::endl;
        logFile << "=== Using parallelizations: " << (useParallel ? "Yes" : "No") << " ===" << std::endl;
        logFile << "=== Data Structure: Struct of Arrays (SoA) ===" << std::endl;
    }
    else
    {
        std::cerr << "Ne mogu otvoriti log.txt za pisanje!" << std::endl;
    }

    std::vector<sf::CircleShape> circles; // Pre-allocate circles
    for (int i = 0; i < n; i++)
    {
        double angle = dist01(rng) * 2 * M_PI;
        double r = dist01(rng) * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        double v = std::sqrt(G_local * mass * n / (r + 50.0));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.3;

        vel.x += (dist01(rng) - 0.5) * v * 0.05;
        vel.y += (dist01(rng) - 0.5) * v * 0.05;

        // 🌟 Populate SoA instead of AoS
        data.m[i] = mass;
        data.posX[i] = pos.x;
        data.posY[i] = pos.y;
        data.velX[i] = vel.x;
        data.velY[i] = vel.y;
        data.accX[i] = 0.0; // Initial acceleration is zero
        data.accY[i] = 0.0;
        data.radius[i] = 8.0;

        // Pre-create circle (only once)
        sf::CircleShape circle(8.0);
        circle.setOrigin(8.0, 8.0);
        circles.push_back(circle);
    }

    sf::View view(sf::Vector2f(0, 0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    double zoomLevel = 1.0;

    // --- SFML tekst (overlay) ---
    const std::vector<std::string> candidates = {
        "/home/fejzullah/Desktop/ETF-master-SA/3_PRS/kod/N-Body Simulation/dejavu-sans/DejaVuSans-Bold.ttf",
        "dejavu-sans/DejaVuSans.ttf",
        "./DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "arial/ARIALBD.TTF",
        "arial/arialbd.ttf",
        "arial.ttf",
        "./arial.ttf"};

    sf::Font font;
    bool fontLoaded = false;
    for (auto &p : candidates)
    {
        std::cout << "Trying: " << p;
        std::ifstream fchk(p, std::ios::binary);
        if (!fchk.is_open())
        {
            std::cout << " -> EXISTS but cannot open (permission?)" << std::endl;
            continue;
        }
        fchk.close();

        // pokuša učitati SFML font
        if (font.loadFromFile(p))
        {
            std::cout << " -> LOADED OK" << std::endl;
            fontLoaded = true;
            break;
        }
        else
        {
            std::cout << " -> loadFromFile FAILED (SFML couldn't create font face)" << std::endl;
        }
    }

    if (!fontLoaded)
    {
        std::cerr << "Font nije učitan. Pokušaj koristeći apsolutnu putanju ili provjeri perms i tip fajla." << std::endl;
    }

    sf::Text overlay;
    if (fontLoaded)
    {
        overlay.setFont(font);
        overlay.setCharacterSize(18); // malo veće slovo
        overlay.setFillColor(sf::Color::Green);
        overlay.setStyle(sf::Text::Bold);
        overlay.setOutlineColor(sf::Color::Black);
        overlay.setOutlineThickness(1.f); // deblja slova izgledaju bolje na crnoj pozadini
        overlay.setPosition(8.f, 8.f);
    }
    else
    {
        std::cerr << "Font nije učitan. Stavi DejaVuSans.ttf ili arial.ttf pored exe-a ili u resources/." << std::endl;
    }

    // --- Pre-allocate pool ONCE (before loop) ---
    int poolSize = n * 16;  // Increased from n*8 for safety with larger body counts
    std::vector<QuadNode> pool(poolSize);

    using Clock = std::chrono::high_resolution_clock;
    int frameCount = 0;
    double elapsedTime = 0.0;

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
                    zoomLevel *= 0.9;
                else
                    zoomLevel *= 1.1;
                view.setSize(WINDOW_SIZE * zoomLevel, WINDOW_SIZE * zoomLevel);
            }
        }

        // --- Physics ---
        auto physicsStart = Clock::now();

        if (!useBarnesHut)
        {
            // 🌟 O(N^2) Loop adapted to SoA
            // Reset acceleration
            for (int i = 0; i < n; ++i)
            {
                data.accX[i] = 0.0;
                data.accY[i] = 0.0;
            }

// Calculate pairwise forces
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i)
            {
                for (int j = i + 1; j < n; ++j)
                {
                    Vec delta(data.posX[j] - data.posX[i], data.posY[j] - data.posY[i]);
                    double dist = delta.len();
                    double minDist = data.radius[i] + data.radius[j];
                    double safeDist = std::max(dist, minDist);
                    double F = (G_local * data.m[j]) / (safeDist * safeDist);
                    Vec acc = delta.norm() * F;

                    // Atomic update needed if parallel (or use a temporary acc array if not OpenMP)
                    // For the sake of simplicity and correctness in a typical OpenMP setup for O(N^2):
                    // Direct access to SoA elements is fine here for i and j as they are independent writes.

                    // Add acceleration to body i
                    data.accX[i] += acc.x;
                    data.accY[i] += acc.y;

                    // Subtract acceleration from body j
                    data.accX[j] -= acc.x;
                    data.accY[j] -= acc.y;
                }
            }
        }
        else
        {
            // 🌟 Barnes-Hut adapted to SoA

            // Track how many nodes were used last iteration
            static int lastPoolIndex = 1;
            
            // Reset pool state - clear only the used nodes from previous iteration
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

            double theta = 1.0;

            if (useParallel)
            {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < (int)data.num_bodies; ++i)
                {
                    // Reset acceleration for this body
                    Vec acc_out = Vec();

                    // Compute force (acceleration)
                    root.computeForce(i, data, theta, acc_out, G_local);

                    // Store result back into SoA
                    data.setAcc(i, acc_out);
                }
            }
            else
            {
                for (size_t i = 0; i < data.num_bodies; ++i)
                {
                    Vec acc_out = Vec();
                    root.computeForce(i, data, theta, acc_out, G_local);
                    data.setAcc(i, acc_out);
                }
            }
        }

        double dt = 0.05;

        // Velocity-Verlet Integration adapted to SoA
        if (useParallel)
        {
// OpenMP parallel loop directly accessing contiguous arrays for maximum cache efficiency
#pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)data.num_bodies; ++i)
            {
                // Update Velocity (Vx += Ax * dt)
                data.velX[i] += data.accX[i] * dt;
                data.velY[i] += data.accY[i] * dt;

                // Update Position (Px += Vx * dt)
                data.posX[i] += data.velX[i] * dt;
                data.posY[i] += data.velY[i] * dt;
            }
        }
        else
        {
            for (size_t i = 0; i < data.num_bodies; ++i)
            {
                // Update Velocity
                data.velX[i] += data.accX[i] * dt;
                data.velY[i] += data.accY[i] * dt;

                // Update Position
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
            // Update position from SoA
            circles[i].setPosition(data.posX[i], data.posY[i]);
            window.draw(circles[i]);
        }

        // --- Performance logging ---
        if (fontLoaded)
        {
            window.setView(window.getDefaultView());
        }

        auto frameEnd = Clock::now();
        std::chrono::duration<double> frameDuration = frameEnd - frameStart;
        std::chrono::duration<double> physicsDuration = physicsEnd - physicsStart;

        frameCount++;
        elapsedTime += frameDuration.count();

        // compute numeric metrics in locals:
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

            std::string line = (written > 0) ? std::string(buf) : std::string("FPS: N/A |Physics: N/A | Total frame: N/A");

            lastLine = line;

            std::cout << lastLine << std::endl;

            if (logFile.is_open())
            {
                logFile << lastLine << std::endl;
                logFile.flush();
            }

            if (fontLoaded)
            {
                overlay.setString(lastLine);
            }

            frameCount = 0;
            elapsedTime = 0.0;
        }

        // CRTAJ OVERLAY
        if (fontLoaded)
        {
            window.setView(window.getDefaultView()); // UI view
            window.draw(overlay);
            window.setView(view); // vrati world view
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