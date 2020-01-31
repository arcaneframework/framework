#pragma once

#include "hypre_backend.h"

#include <ALIEN/Core/Backend/LinearAlgebra.h>

namespace Alien::Hypre {

  using LinearAlgebra = Alien::LinearAlgebra<BackEnd::tag::hypre>;

}