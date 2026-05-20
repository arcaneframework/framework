# Alien API {#aliendoc_api}

[TOC]

## Alien User API, concepts

Alien User API aims at:

- accessing data structure,
- performing *algebraic* operations,
- performing *computer related* operations.

Alien's design allows to create new APIs to give access to previously listed functionalities.

## Alien layout for header files

### Core

```
alien                          # No or few .h at this level (AlienConfig.h)
├── advanced                   # Utilities for end-user API
│   ├── handlers               # IO
│   ├── kernels                # Access to internal kernels
│   └── utils                  # Utilities
│       └── test_framework     # Set up test environment for APIs
├── backend                    # Plug-in API
└── core                       # Keys objects: Mng, Space, Distribution
```

### User API

```
alien
└── api_name                   # Most files at this level
    └── handlers               # Most Builders/Accessors at this level
        └── fs                 # Disk IO
```

### Backend layout

```
alien
└── backend_name
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref aliendoc_devmanual
</span>
<span class="next_section_button">
\ref aliendoc_coding_style
</span>
</div>
