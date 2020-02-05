#pragma once

#include <ALIEN/Core/Backend/BackEnd.h>

namespace Arccore::MessagePassing {
  class IMessagePassingMng;
}

namespace Alien {

  template<class Matrix, class Vector>
  class IInternalLinearSolver;
}

namespace Alien::Hypre {
  class Matrix;

  class Vector;

  class Options;

  extern IInternalLinearSolver<Matrix, Vector> *InternalLinearSolverFactory(const Options& options);
  extern IInternalLinearSolver<Matrix, Vector> *InternalLinearSolverFactory();
}

namespace Alien {

  template<class Matrix, class Vector>
  class IInternalLinearAlgebra;

  extern IInternalLinearAlgebra<Hypre::Matrix, Hypre::Vector> *
  HypreInternalLinearAlgebraFactory();

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
    typedef Hypre::Options options_type;
    typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
    typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;

    static algebra_type *algebra_factory(Arccore::MessagePassing::IMessagePassingMng *p_mng = nullptr) {
      return HypreInternalLinearAlgebraFactory();
    }

    static solver_type *solver_factory(const options_type& options) {
      return Hypre::InternalLinearSolverFactory(options);
    }

    static solver_type *solver_factory() {
      return Hypre::InternalLinearSolverFactory();
    }
    static BackEndId name() { return "hypre"; }
  };

}
