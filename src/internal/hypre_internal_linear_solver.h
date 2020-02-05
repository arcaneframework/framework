#pragma once

#include <memory>

#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Expression/Solver/SolverStats/SolverStater.h>
#include <ALIEN/Core/Backend/IInternalLinearSolverT.h>
#include <ALIEN/Utils/Trace/ObjectWithTrace.h>

#include "hypre_options.h"

namespace Alien::Hypre {

  class Matrix;

  class Vector;

  class InternalLinearSolver
          : public IInternalLinearSolver<Matrix, Vector>, public ObjectWithTrace {
  public:

    typedef SolverStatus Status;

    InternalLinearSolver();

    InternalLinearSolver(const Options& options);

    virtual ~InternalLinearSolver() {}

  public:

    // Nothing to do
    void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng *pm) {}

    bool solve(const Matrix &A, const Vector &b, Vector &x);

    bool hasParallelSupport() const { return true; }

    //! Etat du solveur
    const Status &getStatus() const;

    const SolverStat &getSolverStat() const { return m_stat; }

  private:

    Status m_status;

    Arccore::Real m_init_time;
    Arccore::Real m_total_solve_time;
    Arccore::Integer m_solve_num;
    Arccore::Integer m_total_iter_num;

    SolverStat m_stat;
    Options m_options;

  private:

    void checkError(const Arccore::String &msg, int ierr, int skipError = 0) const;
  };

}