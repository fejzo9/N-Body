#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

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
    Vec &operator+=(const Vec &o) { x += o.x; y += o.y; return *this; }
    Vec &operator-=(const Vec &o) { x -= o.x; y -= o.y; return *this; }
    double len() const { return std::sqrt(x * x + y * y); }
    Vec norm() const { double l = len(); return l == 0 ? Vec(0, 0) : *this / l; }
};

struct Body
{
    double m;
    Vec pos, vel, acc;
    sf::Color color;
    float radius;
};

// -------------------- Barnes-Hut Quadtree (Pool) --------------------
struct QuadNode
{
    double mass = 0;
    Vec centerOfMass;
    Vec topLeft, bottomRight;
    Body* body = nullptr; // pointer to body if leaf
    QuadNode* children[4] = { nullptr, nullptr, nullptr, nullptr }; // contiguous children

    QuadNode() {}
    QuadNode(const Vec& tl, const Vec& br) : topLeft(tl), bottomRight(br) {}

    bool contains(const Vec& p) const
    {
        return p.x >= topLeft.x && p.x <= bottomRight.x &&
               p.y >= topLeft.y && p.y <= bottomRight.y;
    }

    bool isLeaf() const { return children[0] == nullptr; }

    void subdivide(QuadNode* pool, int& poolIndex)
    {
        Vec mid((topLeft.x + bottomRight.x)/2, (topLeft.y + bottomRight.y)/2);
        children[0] = &pool[poolIndex++]; children[0]->topLeft = topLeft; children[0]->bottomRight = mid; // TL
        children[1] = &pool[poolIndex++]; children[1]->topLeft = Vec(mid.x, topLeft.y); children[1]->bottomRight = Vec(bottomRight.x, mid.y); // TR
        children[2] = &pool[poolIndex++]; children[2]->topLeft = Vec(topLeft.x, mid.y); children[2]->bottomRight = Vec(mid.x, bottomRight.y); // BL
        children[3] = &pool[poolIndex++]; children[3]->topLeft = mid; children[3]->bottomRight = bottomRight; // BR
    }

    void insert(Body* b, QuadNode* pool, int& poolIndex)
    {
        if (!contains(b->pos)) return;

        if (!body && isLeaf()) 
        {
            body = b;
            mass = b->m;
            centerOfMass = b->pos;
            return;
        }

        if (isLeaf()) 
        {
            subdivide(pool, poolIndex);
            if (body) 
            {
                for (int i = 0; i < 4; ++i)
                    children[i]->insert(body, pool, poolIndex);
                body = nullptr;
            }
        }

        for (int i = 0; i < 4; ++i)
            children[i]->insert(b, pool, poolIndex);

        // Update center of mass
        mass = 0;
        centerOfMass = Vec(0,0);
        for (int i = 0; i < 4; ++i)
        {
            mass += children[i]->mass;
            centerOfMass += children[i]->centerOfMass * children[i]->mass;
        }
        if (mass > 0) centerOfMass = centerOfMass / mass;
    }

    void computeForce(Body* b, double theta, Vec& acc_out, double G_local)
    {
        if (body == b || mass == 0) return;

        Vec delta = centerOfMass - b->pos;
        double dist = delta.len();
        double width = bottomRight.x - topLeft.x;

        if (isLeaf() || width / dist < theta)
        {
            double minDist = b->radius;
            double safeDist = std::max(dist, minDist);
            double F = (G_local * mass) / (safeDist * safeDist);
            acc_out += delta.norm() * F;
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                children[i]->computeForce(b, theta, acc_out, G_local);
        }
    }
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Chaotic N-Body Simulation");
    window.setFramerateLimit(60);

    bool useBarnesHut = true;

    std::vector<Body> bodies;
    float G_local = 1.f;
    int n = 800;         
    float spread = 400.f; 
    float mass = 1000.f;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist01(0.f, 1.f);

    for (int i = 0; i < n; i++)
    {
        float angle = dist01(rng) * 2 * M_PI;
        float r = dist01(rng) * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        float v = std::sqrt(G_local * mass * n / (r + 50));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.5f;

        vel.x += (dist01(rng)-0.5f) * v * 0.1f;
        vel.y += (dist01(rng)-0.5f) * v * 0.1f;

        bodies.push_back({mass, pos, vel, Vec(), sf::Color(rand() % 255, rand() % 255, rand() % 255), 8.f});
    }

    sf::View view(sf::Vector2f(0,0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    double zoomLevel = 1.0;

    using Clock = std::chrono::high_resolution_clock;
    int frameCount = 0;
    double elapsedTime = 0.0;

    while (window.isOpen())
    {
        auto frameStart = Clock::now();

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta > 0) zoomLevel *= 0.9;
                else zoomLevel *= 1.1;
                view.setSize(WINDOW_SIZE * zoomLevel, WINDOW_SIZE * zoomLevel);
            }
        }

        // --- Physics ---
        auto physicsStart = Clock::now();

        if (!useBarnesHut)
        {
            for (auto &b : bodies) b.acc = Vec();
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
            // Pool allocation
            std::vector<QuadNode> pool(n * 4); // enough nodes
            int poolIndex = 1; // root is at 0
            Vec topLeft(-spread, -spread);
            Vec bottomRight(spread, spread);
            QuadNode& root = pool[0];
            root.topLeft = topLeft;
            root.bottomRight = bottomRight;

            for (auto& b : bodies) root.insert(&b, pool.data(), poolIndex);

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
        window.clear(sf::Color::Black);
        window.setView(view);

        for (auto &b : bodies)
        {
            sf::CircleShape circle(b.radius);
            circle.setOrigin(b.radius, b.radius);
            circle.setFillColor(b.color);
            circle.setPosition(b.pos.x, b.pos.y);
            window.draw(circle);
        }

        window.display();

        // --- Performance logging ---
        auto frameEnd = Clock::now();
        std::chrono::duration<double> frameDuration = frameEnd - frameStart;
        std::chrono::duration<double> physicsDuration = physicsEnd - physicsStart;

        frameCount++;
        elapsedTime += frameDuration.count();

        if (elapsedTime >= 1.0)
        {
            double fps = frameCount / elapsedTime;
            std::cout << "FPS: " << fps 
                      << " | Physics: " << physicsDuration.count() * 1000.0 << " ms"
                      << " | Total frame: " << frameDuration.count() * 1000.0 << " ms" 
                      << std::endl;
            frameCount = 0;
            elapsedTime = 0.0;
        }
    }

    return 0;
}
