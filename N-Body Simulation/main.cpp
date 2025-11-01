#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

const double G = 6.67430e-3; // gravitational constant (reduced for stability)
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
    double len() const { return std::sqrt(x * x + y * y); }
    Vec norm() const
    {
        double l = len();
        return l == 0 ? Vec(0, 0) : *this / l;
    }
    Vec &operator-=(const Vec &o)
    {
        x -= o.x;
        y -= o.y;
        return *this;
    }
};

struct Body
{
    double m;
    Vec pos, vel, acc;
    sf::Color color;
    float radius;
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Chaotic N-Body Simulation");
    window.setFramerateLimit(60);

    std::vector<Body> bodies;
    float G_local = 1.f;
    int n = 100;          // number of bodies
    float spread = 400.f; // initial position spread
    float mass = 1000.f;

    // Initialize bodies
    for (int i = 0; i < n; i++)
    {
        float angle = static_cast<float>(rand()) / RAND_MAX * 2 * M_PI;
        float r = static_cast<float>(rand()) / RAND_MAX * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        float v = sqrt(G_local * mass * n / (r + 50)); // circular velocity approximation
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.5f; // reduce speed to prevent escape

        // Add randomness to velocities
        vel.x += ((float)rand() / RAND_MAX - 0.5f) * v * 0.1f;
        vel.y += ((float)rand() / RAND_MAX - 0.5f) * v * 0.1f;

        bodies.push_back({mass, pos, vel, Vec(), sf::Color(rand() % 255, rand() % 255, rand() % 255), 8.f});
    }

    // Compute accelerations with minimum distance to avoid collisions
    auto compute_accelerations = [&](std::vector<Body> &B)
    {
        for (auto &b : B)
            b.acc = Vec();

        for (size_t i = 0; i < B.size(); ++i)
        {
            for (size_t j = i + 1; j < B.size(); ++j)
            {
                Vec delta = B[j].pos - B[i].pos;
                double dist = delta.len();

                // Clamp minimum distance to avoid infinite force
                double minDist = B[i].radius + B[j].radius;
                double safeDist = std::max(dist, minDist);

                double F = (G_local * B[j].m) / (safeDist * safeDist);
                Vec acc = delta.norm() * F;
                B[i].acc += acc;
                B[j].acc -= acc; // Newton's 3rd law
            }
        }
    };

    sf::View view(sf::Vector2f(0, 0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    double zoomLevel = 1.0;

    while (window.isOpen())
    {
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

        compute_accelerations(bodies);

        double dt = 0.1;
        for (auto &b : bodies)
        {
            b.vel += b.acc * dt;
            b.pos += b.vel * dt;
        }

        window.clear(sf::Color::Black);
        window.setView(view);

        // Draw bodies
        for (auto &b : bodies)
        {
            sf::CircleShape circle(b.radius);
            circle.setOrigin(b.radius, b.radius);
            circle.setFillColor(b.color);
            circle.setPosition(b.pos.x, b.pos.y);
            window.draw(circle);
        }

        window.display();
    }

    return 0;
}
