.. _developer_api:

=========
Alien API
=========

Alien User API, concepts
========================

Alien User API aims at:
    - accessing data structure,
    - performing *algebraic* operations,
    - performing *computer related* operations.

Alien's design allows to create new APIs to give access to previously listed functionalities.


Alien layout for header files
=============================

Core
----

.. code-block:: bash

    alien                          # No or few .h at this level (AlienConfig.h)
    ├── advanced                   # Utilities for end-user API
    │   ├── handlers               # IO
    │   ├── kernels                # Access to internal kernels
    │   └── utils                  # Utilities
    │       └── test_framework     # Set up test environment for APIs
    ├── backend                    # Plug-in API
    └── core                       # Keys objects: Mng, Space, Distribution


User API
--------

.. code-block:: bash

    alien
    └── api_name                   # Most files at this level
        └── handlers               # Most Builders/Accessors at this level
            └── fs                 # Disk IO

Backend layout
--------------

.. code-block:: bash

    alien
    └── backend_name
