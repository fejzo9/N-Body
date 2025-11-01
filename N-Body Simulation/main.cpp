#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>

const double G = 6.67430e-3; // gravitational constant (scaled for stability)
const int WINDOW_SIZE = 1000;

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
    Vec normalized() const
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
    float radius;
    sf::Color color;
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Chaotic N-Body Simulation");
    window.setFramerateLimit(60);

    // Create bodies
    std::vector<Body> bodies;
    int n = 100;
    float spread = 400.f;
    float mass = 1000.f;
    float G_local = 1.f; // for velocity initialization

    for (int i = 0; i < n; i++)
    {
        float angle = static_cast<float>(rand()) / RAND_MAX * 2 * 3.14159265f;
        float r = static_cast<float>(rand()) / RAND_MAX * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        float v = sqrt(G_local * mass * n / (r + 50));
        Vec vel(-pos.y, pos.x);
        vel = vel.normalized() * v * 0.5f;

        // small randomness
        vel.x += ((float)rand() / RAND_MAX - 0.5f) * v * 0.1f;
        vel.y += ((float)rand() / RAND_MAX - 0.5f) * v * 0.1f;

        bodies.push_back({mass, pos, vel, Vec(), 8.f, sf::Color::White});
    }

    sf::View view(sf::Vector2f(0, 0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    double zoomLevel = 1.0;

    sf::Clock clock; // for frame timing

    while (window.isOpen())
    {
        // --- Event handling ---
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

        // --- Time step ---
        sf::Time elapsed = clock.restart();
        double dt = elapsed.asSeconds();

        // --- Compute accelerations ---
        for (size_t i = 0; i < bodies.size(); ++i)
        {
            bodies[i].acc = Vec();
        }

        for (size_t i = 0; i < bodies.size(); ++i)
        {
            for (size_t j = i + 1; j < bodies.size(); ++j)
            {
                Vec delta = bodies[j].pos - bodies[i].pos;
                double dist = delta.len();

                double minDist = bodies[i].radius + bodies[j].radius;
                double safeDist = std::max(dist, minDist);

                double F = (G * bodies[j].m) / (safeDist * safeDist);
                Vec acc = delta.normalized() * F;

                bodies[i].acc += acc;
                bodies[j].acc -= acc; // Newton's 3rd law
            }
        }

        // --- Update positions & velocities ---
        for (auto &b : bodies)
        {
            b.vel += b.acc * dt;
            b.pos += b.vel * dt;
        }

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
        std::cout << "Frame time: " << elapsed.asMilliseconds() << " ms, FPS: " << 1.0 / dt << std::endl;
    }

    return 0;
}
