#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <memory> // for std::unique_ptr

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
    double len() const { return std::sqrt(x * x + y * y); }
    Vec norm() const { double l = len(); return l == 0 ? Vec(0, 0) : *this / l; }
    Vec &operator-=(const Vec &o) { x -= o.x; y -= o.y; return *this; }
};

struct Body
{
    double m;
    Vec pos, vel, acc;
    sf::Color color;
    float radius;
};

// -------------------- Barnes-Hut Quadtree --------------------
struct QuadNode
{
    double mass = 0;
    Vec centerOfMass;
    Vec topLeft, bottomRight;
    std::vector<std::unique_ptr<QuadNode>> children; // 4 quadrants
    Body* body = nullptr; // pointer to body if leaf node

    QuadNode(const Vec& tl, const Vec& br) : topLeft(tl), bottomRight(br) {}

    bool contains(const Vec& p) const
    {
        return p.x >= topLeft.x && p.x <= bottomRight.x &&
               p.y >= topLeft.y && p.y <= bottomRight.y;
    }

    // Check if node is leaf
    bool isLeaf() const { return children.empty(); }

    // Subdivide node
    void subdivide()
    {
        Vec mid((topLeft.x + bottomRight.x)/2, (topLeft.y + bottomRight.y)/2);
        children.push_back(std::make_unique<QuadNode>(topLeft, mid)); // TL
        children.push_back(std::make_unique<QuadNode>(Vec(mid.x, topLeft.y), Vec(bottomRight.x, mid.y))); // TR
        children.push_back(std::make_unique<QuadNode>(Vec(topLeft.x, mid.y), Vec(mid.x, bottomRight.y))); // BL
        children.push_back(std::make_unique<QuadNode>(mid, bottomRight)); // BR
    }

    void insert(Body* b)
    {
        if (!contains(b->pos)) return; // if body not is out of bounds don't insert

        if (!body && isLeaf()) 
        {
            body = b;
            mass = b->m;
            centerOfMass = b->pos;
            return;
        }

        if (isLeaf()) 
        {
            subdivide();
            if (body) 
            {
                for (auto& child : children)
                    child->insert(body);
                body = nullptr;
            }
        }

        for (auto& child : children)
            child->insert(b);

        // Update center of mass
        mass = 0;
        centerOfMass = Vec(0,0);
        for (auto& child : children)
        {
            mass += child->mass;
            centerOfMass += child->centerOfMass * child->mass;
        }
        if (mass > 0) centerOfMass = centerOfMass / mass;
    }

    // Compute force on body using BH approximation
    void computeForce(Body* b, double theta, Vec& acc_out, double G_local)
    {
        if (body == b || mass == 0) return; // Skip self or empty node

        Vec delta = centerOfMass - b->pos;
        double dist = delta.len();
        double width = bottomRight.x - topLeft.x;

        if (isLeaf() || width / dist < theta) // If sufficiently far or leaf
        {
            double minDist = b->radius;
            double safeDist = std::max(dist, minDist);
            double F = (G_local * mass) / (safeDist * safeDist);
            acc_out += delta.norm() * F;
        }
        else
        {
            for (auto& child : children)
                child->computeForce(b, theta, acc_out, G_local);
        }
    }
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Chaotic N-Body Simulation");
    window.setFramerateLimit(60);

    bool useBarnesHut = true; // <---- switch between naive and BH

    std::vector<Body> bodies;
    float G_local = 1.f;
    int n = 1000;         
    float spread = 400.f; 
    float mass = 1000.f;

    for (int i = 0; i < n; i++)
    {
        float angle = static_cast<float>(rand()) / RAND_MAX * 2 * M_PI;
        float r = static_cast<float>(rand()) / RAND_MAX * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        float v = sqrt(G_local * mass * n / (r + 50));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.5f;

        vel.x += ((float)rand() / RAND_MAX - 0.5f) * v * 0.1f;
        vel.y += ((float)rand() / RAND_MAX - 0.5f) * v * 0.1f;

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
            // Naive O(n^2)
            for (auto &b : bodies) b.acc = Vec();
            for (size_t i = 0; i < bodies.size(); ++i)
            {
                for (size_t j = i + 1; j < bodies.size(); ++j)
                {
                    Vec delta = bodies[j].pos - bodies[i].pos;
                    double dist = delta.len();

                    // Clamp minimum distance to avoid infinite force
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
            // Barnes-Hut
            Vec topLeft(-spread, -spread);
            Vec bottomRight(spread, spread);
            QuadNode root(topLeft, bottomRight);
            for (auto& b : bodies) root.insert(&b);
            double theta = 0.5; // BH opening angle
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
