// -*- C++ -*-
#ifndef ALIEN_MCGIMPL_MCGCOMPOSITEVECTOR_H
#define ALIEN_MCGIMPL_MCGCOMPOSITEVECTOR_H

#include <alien/utils/Precomp.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>

#include "alien/kernels/mcg/MCGPrecomp.h"

BEGIN_MCGINTERNAL_NAMESPACE

class CompositeVectorInternal;

END_MCGINTERNAL_NAMESPACE

namespace Alien {

class MCGCompositeVector : public IVectorImpl
{
 public:
  typedef MCGInternal::CompositeVectorInternal VectorInternal;

 public:
  MCGCompositeVector(const MultiVectorImpl* multi_impl);

  virtual ~MCGCompositeVector();

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void allocate();

  void free() {}
  void clear() {}

 public:
  void setValues(const int part, const double* values);
  void getValues(const int part, double* values) const;

 public:
  VectorInternal* internal() { return m_internal; }
  const VectorInternal* internal() const { return m_internal; }

  void update(const MCGCompositeVector& v);

 private:
  bool assemble();

 private:
  VectorInternal* m_internal = nullptr;
};

} // namespace Alien

#endif /* ALIEN_MCGIMPL_MCGCOMPOSITEVECTOR_H */
