#include <SFML/Graphics.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>
#include "Simulation.h"

// Globalne varijable za sinhronizaciju između simulacionog i render thread-a
std::atomic<bool> paused(false);        // Da li je simulacija pauzirana
std::atomic<int> tick(0);               // Brojač iteracija simulacije
std::atomic<uint64_t> new_seed(0);      // Novi seed za reset simulacije
std::atomic<bool> seed_changed(false);  // Flag da li je seed promijenjen

std::mutex bodies_mutex;                // Mutex za thread-safe pristup tijelima
std::vector<Body> shared_bodies;        // Kopija tijela za rendering

const uint64_t START_SEED = 3;          // Početni seed

// Thread funkcija za simulaciju
void simulation_thread() {
    Simulation sim(START_SEED);
    
    while (true) {
        // Provjeri da li treba resetovati simulaciju sa novim seed-om
        if (seed_changed.load()) {
            uint64_t seed = new_seed.load();
            sim = Simulation(seed);
            tick.store(0);
            seed_changed.store(false);
        }

        // Ako je simulacija pauzirana, sačekaj
        if (paused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }

        // Izvrši jedan korak simulacije
        sim.update();
        
        // Thread-safe kopiranje tijela za rendering
        {
            std::lock_guard<std::mutex> lock(bodies_mutex);
            shared_bodies = sim.bodies;
        }
        
        tick.fetch_add(1);

        // Ograniči brzinu simulacije (opciono)
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

int main() {
    const int WINDOW_WIDTH = 900;
    const int WINDOW_HEIGHT = 900;
    
    // Kreiraj prozor
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "N-Body Simulation");
    window.setFramerateLimit(60);

    // Pokreni simulacioni thread
    std::thread sim_thread(simulation_thread);
    sim_thread.detach();

    // View parametri za zoom i pan
    float view_scale = 2.0f;
    sf::Vector2f view_pos(0.0f, 0.0f);
    
    // Mouse kontrole
    sf::Vector2i last_mouse_pos;
    bool is_dragging = false;

    // UI parametri
    uint64_t current_seed = START_SEED;

    // Glavni rendering loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            
            // Tastatura kontrole
            if (event.type == sf::Event::KeyPressed) {
                // Space - pauza/resume
                if (event.key.code == sf::Keyboard::Space) {
                    paused.store(!paused.load());
                }
                // R - randomize seed
                if (event.key.code == sf::Keyboard::R) {
                    current_seed = std::random_device{}();
                    new_seed.store(current_seed);
                    seed_changed.store(true);
                }
                // N - sljedeći seed
                if (event.key.code == sf::Keyboard::N) {
                    current_seed++;
                    new_seed.store(current_seed);
                    seed_changed.store(true);
                }
            }
            
            // Mouse wheel - zoom
            if (event.type == sf::Event::MouseWheelScrolled) {
                float zoom_factor = (event.mouseWheelScroll.delta > 0) ? 0.9f : 1.1f;
                
                // Zoom prema poziciji miša
                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
                sf::Vector2f world_pos_before = window.mapPixelToCoords(mouse_pos);
                
                view_scale *= zoom_factor;
                
                // Ažuriraj view
                sf::View view = window.getView();
                view.setSize(view_scale * WINDOW_WIDTH, view_scale * WINDOW_HEIGHT);
                view.setCenter(view_pos);
                window.setView(view);
                
                sf::Vector2f world_pos_after = window.mapPixelToCoords(mouse_pos);
                view_pos += world_pos_before - world_pos_after;
            }
            
            // Mouse drag - pan
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Right) {
                    is_dragging = true;
                    last_mouse_pos = sf::Mouse::getPosition(window);
                }
            }
            
            if (event.type == sf::Event::MouseButtonReleased) {
                if (event.mouseButton.button == sf::Mouse::Right) {
                    is_dragging = false;
                }
            }
        }
        
        // Mouse dragging za pomeranje view-a
        if (is_dragging) {
            sf::Vector2i current_mouse = sf::Mouse::getPosition(window);
            sf::Vector2i delta = current_mouse - last_mouse_pos;
            
            view_pos.x -= delta.x * view_scale;
            view_pos.y -= delta.y * view_scale;
            
            last_mouse_pos = current_mouse;
        }

        // Postavi view
        sf::View view = window.getView();
        view.setSize(view_scale * WINDOW_WIDTH, view_scale * WINDOW_HEIGHT);
        view.setCenter(view_pos);
        window.setView(view);

        // Očisti prozor
        window.clear(sf::Color::Black);

        // Nacrtaj tijela
        {
            std::lock_guard<std::mutex> lock(bodies_mutex);
            
            for (const auto& body : shared_bodies) {
                // Konvertuj poziciju u ekranske koordinate
                float radius = 0.05f; // Radijus kruga
                sf::CircleShape circle(radius);
                circle.setOrigin(radius, radius);
                circle.setPosition(body.pos.x, body.pos.y);
                circle.setFillColor(sf::Color::White);
                
                window.draw(circle);
            }
        }

        // Prikaži informacije (jednostavno, bez GUI biblioteke)
        sf::View ui_view = window.getDefaultView();
        window.setView(ui_view);
        
        // Info tekst - jednostavan prikaz
        // (Za pravi GUI trebalo bi koristiti ImGui ili sličnu biblioteku)
        
        window.display();
    }

    return 0;
}

/* 
KONTROLE:
- Space: Pauziraj/Nastavi simulaciju
- R: Nasumičan novi seed
- N: Sljedeći seed (inkrementuj)
- Scroll miša: Zoom in/out
- Desni klik i drag: Pomjeraj view
- ESC: Zatvori aplikaciju (ako dodaš u event handling)
*/