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

    void subdivide(QuadNode* pool, int& poolIndex, int poolSize)
    {
        // zaštita od overflow-a pool-a
        if (poolIndex + 4 >= poolSize) return;
        Vec mid((topLeft.x + bottomRight.x)/2, (topLeft.y + bottomRight.y)/2);
        children[0] = &pool[poolIndex++]; children[0]->topLeft = topLeft; children[0]->bottomRight = mid; // TL
        children[1] = &pool[poolIndex++]; children[1]->topLeft = Vec(mid.x, topLeft.y); children[1]->bottomRight = Vec(bottomRight.x, mid.y); // TR
        children[2] = &pool[poolIndex++]; children[2]->topLeft = Vec(topLeft.x, mid.y); children[2]->bottomRight = Vec(mid.x, bottomRight.y); // BL
        children[3] = &pool[poolIndex++]; children[3]->topLeft = mid; children[3]->bottomRight = bottomRight; // BR
    }

    void insert(Body* b, QuadNode* pool, int& poolIndex, int poolSize)
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
        const double eps = 1e-6; // zaštita za dijeljenje i softening
        double safeDistForTheta = dist + eps;
        double width = bottomRight.x - topLeft.x;

        if (isLeaf() || (width / safeDistForTheta) < theta)
        {
            double minDist = b->radius;
            double safeDist = std::max(dist, minDist);
            // softening: dodamo malu vrijednost u nazivnik da spriječimo ekstremne akceleracije
            double soft = 0.001;
            double F = (G_local * mass) / (safeDist * safeDist + soft*soft);
            if (dist > eps) acc_out += delta.norm() * F;
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                if (children[i]) children[i]->computeForce(b, theta, acc_out, G_local);
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

    std::vector<Body> bodies;
    double G_local = 1.0;
    int n = 10000;         
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

    std::vector<sf::CircleShape> circles; // Pre-allocate circles
    for (int i = 0; i < n; i++)
    {
        double angle = dist01(rng) * 2 * M_PI;
        double r = dist01(rng) * spread;
        Vec pos(r * cos(angle), r * sin(angle));

        double v = std::sqrt(G_local * mass * n / (r + 50.0));
        Vec vel(-pos.y, pos.x);
        vel = vel.norm() * v * 0.5;

        vel.x += (dist01(rng)-0.5) * v * 0.1;
        vel.y += (dist01(rng)-0.5) * v * 0.1;

        bodies.push_back({mass, pos, vel, Vec(), sf::Color(rand() % 255, rand() % 255, rand() % 255), 8.0});
        
        // Pre-create circle (only once)
        sf::CircleShape circle(8.0);
        circle.setOrigin(8.0, 8.0);
        circle.setFillColor(bodies[i].color);
        circles.push_back(circle);
    }

    sf::View view(sf::Vector2f(0,0), sf::Vector2f(WINDOW_SIZE, WINDOW_SIZE));
    double zoomLevel = 1.0;

    // --- SFML tekst (overlay) ---

    // pokušava nekoliko uobičajenih lokacija/imena fontova
    // const char* fontCandidates[] = {
        //     "dejavu-sans/DejaVuSans-Bold.ttf",
        //     "./DejaVuSans.ttf",
        //     "arial/ARIALBD.TTF",
        //     "./arial.ttf"
        // };
        const std::vector<std::string> candidates = {
            "/home/fejzullah/Desktop/ETF-master-SA/3_PRS/kod/N-Body Simulation/dejavu-sans/DejaVuSans-Bold.ttf",
            "dejavu-sans/DejaVuSans.ttf",
            "./DejaVuSans.ttf",
            "DejaVuSans.ttf",
            "arial/ARIALBD.TTF",
            "arial/arialbd.ttf",
            "arial.ttf",
            "./arial.ttf"
        };
        
    sf::Font font;
    bool fontLoaded = false;
    for (auto &p : candidates)
    {
         std::cout << "Trying: " << p;
        std::ifstream fchk(p, std::ios::binary);
    if (!fchk.is_open()) {
        std::cout << " -> EXISTS but cannot open (permission?)" << std::endl;
        continue;
    }
    fchk.close();

    // pokuša učitati SFML font
    if (font.loadFromFile(p)) {
        std::cout << " -> LOADED OK" << std::endl;
        fontLoaded = true;
        break;
    } else {
        std::cout << " -> loadFromFile FAILED (SFML couldn't create font face)" << std::endl;
        }
    }

    if (!fontLoaded) {
        std::cerr << "Font nije učitan. Pokušaj koristeći apsolutnu putanju (/full/path/to/DejaVuSans.ttf) ili provjeri perms i tip fajla." << std::endl;
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
    int poolSize = n * 8;
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
            // Reset pool state and reuse (memset clears all node memory to zero)
            std::memset(pool.data(), 0, poolSize * sizeof(QuadNode));
            int poolIndex = 1;
            
            Vec topLeft(-spread, -spread);
            Vec bottomRight(spread, spread);
            QuadNode& root = pool[0];
            root.topLeft = topLeft;
            root.bottomRight = bottomRight;

            for (auto& b : bodies) root.insert(&b, pool.data(), poolIndex, poolSize);

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

        for (size_t i = 0; i < bodies.size(); ++i)
        {
            // Just update position, don't recreate
            circles[i].setPosition(bodies[i].pos.x, bodies[i].pos.y);
            window.draw(circles[i]);
        }

        // Prikažemo overlay u default view (UI), da ne zumira sa scenom:
        if (fontLoaded)
        {
            window.setView(window.getDefaultView());
        }

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
        if (elapsedTime > 0.0) fpsValue = frameCount / elapsedTime;

        // format line robustly:
        char buf[256];

        // interval u sekundama za update overlaya i loga
        static const double logInterval = 1.0; // možemo promjeniti na 0.5 ili 2.0, itd.
        static std::string lastLine;            // zadržava posljednju "full" liniju

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

            // ispis na ekran preko SFML-a (ako imamo font)
            if (fontLoaded)
            {
                overlay.setString(lastLine);
            }

            // reset brojača
            frameCount = 0;
            elapsedTime = 0.0;
        }

        // CRTAJ OVERLAY U DEFAULT VIEW (UI) — nakon što si prethodno nacrtao svijet
        if (fontLoaded)
        {
            window.setView(window.getDefaultView()); // UI view
            window.draw(overlay);
            window.setView(view); // vrati world view (ako će se nakon toga crtati još nešto u world koordinatama)
        }
        // TODO: optimize windows display calls
        window.display();
    }

    // pri kraju sesije zabilježimo kraj simulacije
    if (logFile.is_open())
    {
        logFile << "=== Simulation ended: " << currentDateTimeString() << " ===" << std::endl << std::endl;
        logFile.close();
    }

    return 0;
}
