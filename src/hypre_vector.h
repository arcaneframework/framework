#pragma once

#include <ALIEN/Core/Impl/IVectorImpl.h>
#include <ALIEN/Distribution/VectorDistribution.h>

namespace Alien::Hypre::Internal {
  class VectorInternal;
}

namespace Alien::Hypre {

  class VectorInternal;

  class Vector : public IVectorImpl {
  public:

    typedef Internal::VectorInternal VectorInternal;

  public:

    Vector(const MultiVectorImpl *multi_impl);

    virtual ~Vector();

  public:
    void init(const VectorDistribution &dist, const bool need_allocate);

    void allocate();

    void free() {}

    void clear() {}

  public:
    bool setValues(const int nrow,
                   const double *values);

    bool setValues(const int nrow,
                   const int *rows,
                   const double *values);

  public:
    bool getValues(const int nrow,
                   const int *rows,
                   double *values) const;

    bool getValues(const int nrow,
                   double *values) const;

  public:

    // Méthodes restreintes à usage interne de l'implémentation HYPRE
    VectorInternal *internal() { return m_internal; }

    const VectorInternal *internal() const { return m_internal; }

    // These functions should be removed when the relevant Converters will be implemented
    void update(const Vector &v);

  private:

    bool assemble();

  private:
    VectorInternal *m_internal;
    Arccore::Integer m_block_size;
    Arccore::Integer m_offset;
    Arccore::UniqueArray<Arccore::Integer> m_rows;
  };

}
