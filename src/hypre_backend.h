#pragma once

#include <ALIEN/Core/Backend/BackEnd.h>

class IOptionsHypreSolver;

namespace Arccore::MessagePassing {
  class IMessagePassingMng;
}

namespace Alien::Hypre {
  class Matrix;
  class Vector;
  class IOptions;
}

namespace Alien {

  template<class Matrix, class Vector>
  class IInternalLinearAlgebra;

  template<class Matrix, class Vector>
  class IInternalLinearSolver;

  extern IInternalLinearAlgebra<Hypre::Matrix, Hypre::Vector> *
  HypreInternalLinearAlgebraFactory();

  extern IInternalLinearSolver<Hypre::Matrix, Hypre::Vector> *
  HypreInternalLinearSolverFactory(Arccore::MessagePassing::IMessagePassingMng *p_mng, Hypre::IOptions *options);

  namespace BackEnd {
    namespace tag {
      struct hypre {
      };
    }
  }

  template<>
  struct AlgebraTraits<BackEnd::tag::hypre> {
    typedef Hypre::Matrix matrix_type;
    typedef Hypre::Vector vector_type;
    typedef Hypre::IOptions options_type;
    typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
    typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;

    static algebra_type *algebra_factory(Arccore::MessagePassing::IMessagePassingMng *p_mng = nullptr) {
      return HypreInternalLinearAlgebraFactory();
    }

    static solver_type *solver_factory(Arccore::MessagePassing::IMessagePassingMng *p_mng, options_type *options) {
      return HypreInternalLinearSolverFactory(p_mng, options);
    }

    static BackEndId name() { return "hypre"; }
  };

}
