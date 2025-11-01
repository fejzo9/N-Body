# N-Body Simulation

Gravitaciona simulacija N-tijela implementirana u C++ sa SFML bibliotekom za vizualizaciju.

# Opis
Urađena je simulacija gravitacione interakcije između N nebeskih tijela koristeći Newton-ove zakone gravitacije. Tijela se kreću pod uticajem međusobnih gravitacionih sila.

# Struktura projekta
```
kod/
├── N-Body Simulation/
│   ├── Vec2.h          # 2D vektor matematika
│   ├── Body.h          # Klasa za nebesko tijelo
│   ├── Simulation.h    # Fizika i gravitacione kalkulacije
│   └── main.cpp        # Glavna aplikacija, rendering i paralelizacija
├── bin/
│   └── main            # Izvršni fajl (kompajliran)
└── .vscode/
    ├── tasks.json      # Build konfiguracija
    └── launch.json     # Debug konfiguracija
```
# Algoritam
Simulacija koristi **N² algoritam** za gravitacione interakcije (između svih parova tijela)
