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
#include <algorithm> // Za std::max

// Konstante
const double G = 6.67430e-3;
// Originalna konstanta WINDOW_SIZE je irelevantna za fiziku pa je izbacujemo.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------- Strukture za fiziku --------------------
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

struct Body
{
    double m;
    Vec pos, vel, acc;
    float radius;
};

// -------------------- Barnes-Hut Quadtree (Pool) --------------------
struct QuadNode
{
    double mass = 0;
    Vec centerOfMass;
    Vec topLeft, bottomRight;
    Body *body = nullptr;
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

    void insert(Body *b, QuadNode *pool, int &poolIndex, int poolSize)
    {
        if (!contains(b->pos))
            return;

        if (!body && isLeaf())
        {
            body = b;
            mass = b->m;
            centerOfMass = b->pos;
            return;
        }

        if (isLeaf())
        {
            subdivide(pool, poolIndex, poolSize);
            if (body)
            {
                for (int i = 0; i < 4; ++i)
                    children[i]->insert(body, pool, poolIndex, poolSize);
                body = nullptr;
            }
        }

        for (int i = 0; i < 4; ++i)
            children[i]->insert(b, pool, poolIndex, poolSize);

        // Update center of mass
        mass = 0;
        centerOfMass = Vec(0, 0);
        for (int i = 0; i < 4; ++i)
        {
            mass += children[i]->mass;
            centerOfMass += children[i]->centerOfMass * children[i]->mass;
        }
        if (mass > 0)
            centerOfMass = centerOfMass / mass;
    }

    void computeForce(Body *b, double theta, Vec &acc_out, double G_local)
    {
        if (body == b || mass == 0)
            return;

        Vec delta = centerOfMass - b->pos;
        double dist = delta.len();
        const double eps = 1e-6; // zaštita za dijeljenje i softening
        double safeDistForTheta = dist + eps;
        double width = bottomRight.x - topLeft.x;

        if (isLeaf() || (width / safeDistForTheta) < theta)
        {
            double minDist = b->radius;
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
                    children[i]->computeForce(b, theta, acc_out, G_local);
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
    // --- Inicijalizacija i postavljanje ---
    bool useBarnesHut = true;

    std::vector<Body> bodies;
    double G_local = 1.0;
    int n = 1000;
    double spread = 400.0;
    double mass = 1000.0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    // --- LOG fajl (append) ---
    std::ofstream logFile("log.txt", std::ios::app);
    if (logFile.is_open())
    {
        logFile << "=== Simulation started: " << currentDateTimeString() << " ===" << std::endl;
        logFile << "=== Number of bodies: " << n << " ===" << std::endl;
    }
    else
    {
        std::cerr << "Ne mogu otvoriti log.txt za pisanje!" << std::endl;
    }

    // Uklanjamo pre-alokaciju sf::CircleShape-ova

    for (int i = 0; i < n; i++)
    {
        double angle = dist01(rng) * 2 * M_PI;
        double r = dist01(rng) * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        double v = std::sqrt(G_local * mass * n / (r + 50.0));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.5;

        vel.x += (dist01(rng) - 0.5) * v * 0.1;
        vel.y += (dist01(rng) - 0.5) * v * 0.1;

        bodies.push_back({mass, pos, vel, Vec(), 8.0});
    }

    // Uklanjamo SFML view i zoom logiku

    // Uklanjamo SFML font i tekst logiku

    // --- Pre-allocate pool ONCE (before loop) ---
    int poolSize = n * 8;
    std::vector<QuadNode> pool(poolSize);

    using Clock = std::chrono::high_resolution_clock;
    int frameCount = 0;
    double elapsedTime = 0.0;

    // Glavna petlja simulacije (Beskonačna petlja, kao što je bilo while(window.isOpen()))
    while (true)
    {
        auto frameStart = Clock::now();

        // Uklanjamo SFML Event handling (zatvaranje, zoom)

        // --- Physics ---
        auto physicsStart = Clock::now();

        if (!useBarnesHut)
        {
            for (auto &b : bodies)
                b.acc = Vec();
            for (size_t i = 0; i < bodies.size(); ++i)
            {
                for (size_t j = i + 1; j < bodies.size(); ++j)
                {
                    Vec delta = bodies[j].pos - bodies[i].pos;
                    double dist = delta.len();
                    double minDist = bodies[i].radius + bodies[j].radius;
                    double safeDist = std::max(dist, minDist);
                    double F = (G_local * bodies[j].m) / (safeDist * safeDist);
                    Vec acc = delta.norm() * F;
                    bodies[i].acc += acc;
                    bodies[j].acc -= acc;
                }
            }
        }
        else
        {
            // Reset pool state and reuse (memset clears all node memory to zero)
            std::memset(pool.data(), 0, poolSize * sizeof(QuadNode));
            int poolIndex = 1;

            Vec topLeft(-spread, -spread);
            Vec bottomRight(spread, spread);
            QuadNode &root = pool[0];
            root.topLeft = topLeft;
            root.bottomRight = bottomRight;

            for (auto &b : bodies)
                root.insert(&b, pool.data(), poolIndex, poolSize);

            double theta = 0.5;
            for (auto &b : bodies)
            {
                b.acc = Vec();
                root.computeForce(&b, theta, b.acc, G_local);
            }
        }

        double dt = 0.1;
        for (auto &b : bodies)
        {
            b.vel += b.acc * dt;
            b.pos += b.vel * dt;
        }

        auto physicsEnd = Clock::now();

        // --- Rendering ---
        // Uklanjamo window.clear() i crtanje tijela

        // Uklanjamo SFML crtanje overlay-a

        // --- Performance logging ---
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

        // format line robustly:
        char buf[256];

        // interval u sekundama za update overlaya i loga
        static const double logInterval = 1.0;
        static std::string lastLine;

        if (elapsedTime >= logInterval)
        {
            // formiraj liniju jednom, deterministički
            int written = std::snprintf(buf, sizeof(buf),
                                        "FPS: %.2f | Physics: %.2f ms | Total frame: %.2f ms",
                                        std::isfinite(fpsValue) ? fpsValue : 0.0,
                                        std::isfinite(physicsMs) ? physicsMs : 0.0,
                                        std::isfinite(frameMs) ? frameMs : 0.0);

            std::string line = (written > 0) ? std::string(buf) : std::string("FPS: N/A |Physics: N/A | Total frame: N/A");

            // spremimo u lastLine i ispišemo
            lastLine = line;

            // ispis u konzolu
            std::cout << lastLine << std::endl;

            // append u log fajl (sa timestampom)
            if (logFile.is_open())
            {
                logFile << lastLine << std::endl;
                logFile.flush();
            }

            // Uklanjamo SFML postavljanje stringa
            
            // reset brojača
            frameCount = 0;
            elapsedTime = 0.0;
        }

        // Uklanjamo window.display()
    }

    // pri kraju sesije zabilježimo kraj simulacije
    if (logFile.is_open())
    {
        logFile << "=== Simulation ended: " << currentDateTimeString() << " ===" << std::endl
                << std::endl;
        logFile.close();
    }

    return 0;
}