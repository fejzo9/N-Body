#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm> // ...added...

const float PI = 3.14159265f;
const int WINDOW_SIZE = 1000;
// const float AU = 40.0f;               // removed fixed AU
const float SPEED_MULTIPLIER = 0.01f; // Speed up animation
const float ORBIT_MARGIN = 6.0f;      // minimum gap between sun/planet orbit

struct Planet
{
    std::string name;
    float orbitalRadius; // Distance from sun (AU)
    float orbitalPeriod; // Orbital period (Earth years)
    float radius;        // Planet size for rendering (pixels)
    sf::Color color;
    float currentAngle; // Current position in orbit (radians)
};

class SolarSystem
{
private:
    std::vector<Planet> planets;
    float sunRadius = 20.0f;

public:
    SolarSystem()
    {
        // Mercury
        planets.push_back({"Mercury", 0.387f, 0.241f, 4, sf::Color::White, 0.0f});

        // Venus
        planets.push_back({"Venus", 0.723f, 0.615f, 7, sf::Color::Cyan, 0.0f});

        // Earth
        planets.push_back({"Earth", 1.0f, 1.0f, 8, sf::Color::Blue, 0.0f});

        // Mars
        planets.push_back({"Mars", 1.524f, 1.881f, 5, sf::Color::Red, 0.0f});

        // Jupiter
        planets.push_back({"Jupiter", 5.203f, 11.86f, 20, sf::Color::Magenta, 0.0f});

        // Saturn
        planets.push_back({"Saturn", 9.537f, 29.46f, 18, sf::Color::Yellow, 0.0f});
    }

    void update()
    {
        // Update each planet's angle based on Kepler's laws
        for (auto &planet : planets)
        {
            // Angular velocity (radians per frame)
            // Kepler's 3rd law: T^2 proportional to a^3
            // Angular velocity = 2*PI / Period
            float angularVelocity = (2.0f * PI / planet.orbitalPeriod) * SPEED_MULTIPLIER;
            planet.currentAngle += angularVelocity;

            // Keep angle in [0, 2*PI]
            if (planet.currentAngle > 2.0f * PI)
                planet.currentAngle -= 2.0f * PI;
        }
    }

    void render(sf::RenderWindow &window)
    {
        window.clear(sf::Color::Black);

        // Determine view center and half-size (world coordinates)
        sf::Vector2f viewCenter = window.getView().getCenter();
        sf::Vector2f viewSize = window.getView().getSize();
        float halfView = std::min(viewSize.x, viewSize.y) * 0.5f;

        // Find largest orbital AU and largest planet radius to compute scale
        float maxOrbitalAU = 0.0f;
        float maxPlanetRadius = 0.0f;
        for (const auto &p : planets)
        {
            maxOrbitalAU = std::max(maxOrbitalAU, p.orbitalRadius);
            maxPlanetRadius = std::max(maxPlanetRadius, p.radius);
        }

        // Compute AU scale so the outermost planet fits inside the view (with some margin)
        float available = halfView - sunRadius - ORBIT_MARGIN - maxPlanetRadius;
        if (available < 10.0f) // safety clamp
            available = 10.0f;
        float AU_scale = (maxOrbitalAU > 0.0f) ? (available / maxOrbitalAU) : 1.0f;

        int centerX = static_cast<int>(viewCenter.x);
        int centerY = static_cast<int>(viewCenter.y);

        // Draw Sun
        sf::CircleShape sun(sunRadius);
        sun.setPosition(centerX - sunRadius, centerY - sunRadius);
        sun.setFillColor(sf::Color::Yellow);
        window.draw(sun);

        // Draw orbital paths (optional guides) and planets using computed radii
        for (const auto &planet : planets)
        {
            float orbitRadius = planet.orbitalRadius * AU_scale;

            // Ensure planet is not orbiting inside the sun
            float minOrbit = sunRadius + planet.radius + ORBIT_MARGIN;
            if (orbitRadius < minOrbit)
                orbitRadius = minOrbit;

            // Orbit guide
            sf::CircleShape orbit(orbitRadius);
            orbit.setPosition(centerX - orbitRadius, centerY - orbitRadius);
            orbit.setFillColor(sf::Color::Transparent);
            orbit.setOutlineThickness(1.0f);
            orbit.setOutlineColor(sf::Color::White);
            window.draw(orbit);

            // Planet position
            float x = centerX + orbitRadius * std::cos(planet.currentAngle);
            float y = centerY + orbitRadius * std::sin(planet.currentAngle);

            sf::CircleShape circle(planet.radius);
            circle.setPosition(x - planet.radius, y - planet.radius);
            circle.setFillColor(planet.color);
            window.draw(circle);
        }

        window.display();
    }
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Solar System - Kepler's Laws");
    window.setFramerateLimit(60);

    SolarSystem system;

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

        window.setView(view);
        system.update();
        system.render(window);
    }

    return 0;
}
