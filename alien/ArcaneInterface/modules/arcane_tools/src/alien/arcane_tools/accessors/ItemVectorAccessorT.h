#ifndef ALIEN_ACCESSOR_ITEMVECTORACCESSORT_H
#define ALIEN_ACCESSOR_ITEMVECTORACCESSORT_H

/*---------------------------------------------------------------------------*/

#ifdef ALIEN_INCLUDE_TEMPLATE_IN_CC
// Si on est en debug ou qu'on ne souhaite pas l'inlining, VectorAccessorT
// est inclus dans VectorAccessor.cc
#ifndef ALIEN_ACCESSOR_ITEMVECTORACCESSOR_CC
#error "This file must be used by inclusion in VectorAccessor.cc file"
#endif /* ALIEN_ACCESSOR_VECTORACCESSOR_CC */
#else /*ALIEN_INCLUDE_TEMPLATE_IN_CC */
// Autrement, VectorAccessorT est inclus dans VectorAccessor.h
#ifndef ALIEN_ACCESSOR_ITEMVECTORACCESSOR_H
#error "This file must be used by inclusion in VectorAccessor.h file"
#endif /* ALIEN_ACCESSOR_VECTORACCESSOR_H */
#endif /* ALIEN_INCLUDE_TEMPLATE_IN_CC */

// Tout autre inclusion créera une erreur de compilation

/*---------------------------------------------------------------------------*/

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  ItemVectorAccessorT<ValueT>::ItemVectorAccessorT(IVector& vector, bool update)
  : Common::VectorWriterBaseT<ValueT>(vector, update)
  , m_space(vector.space())
  , m_distribution(vector.impl()->distribution())
  , m_block(vector.impl()->block())
  , m_vblock(vector.impl()->vblock())
  {
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  typename ItemVectorAccessorT<ValueT>::VectorElement ItemVectorAccessorT<ValueT>::
  operator()(const IIndexManager::Entry& entry,
      [[maybe_unused]] typename ItemVectorAccessorT<ValueT>::eSubBlockExtractingPolicyType policy
          )
  {
    if (m_vblock)
      // XT (27/07/2016) : This is just to allow compilation
      return VectorElement(*this, m_block, entry, this->m_values);
    /*
    return VectorElement(*this,
                         m_vblock,
                         entry,
                         this->m_values,
       this->m_vector_impl.vblockImpl().offsetOfLocalIndex(),
                         policy==FirstContiguousIndexes);
    */
    else
      return VectorElement(*this, m_block, entry, this->m_values);
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  ItemVectorAccessorT<ValueT>::VectorElement::VectorElement(
      ItemVectorAccessorT<ValueT>& accessor, const Block* block,
      const IIndexManager::Entry& entry, Arccore::ArrayView<ValueT> values)
  : m_entry(entry)
  , m_main_accessor(accessor)
  , m_values(values)
  , m_first_contiguous(true)
  , m_block(block)
  , m_vblock(nullptr)
  {
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  ItemVectorAccessorT<ValueT>::VectorElement::VectorElement(
      ItemVectorAccessorT<ValueT>& accessor, const VBlock* block,
      const IIndexManager::Entry& entry, Arccore::ArrayView<ValueT> values,
      Arccore::ConstArrayView<Arccore::Integer> values_ptr,
      bool first_contiguous_indexes_policy)
  : m_entry(entry)
  , m_main_accessor(accessor)
  , m_values(values)
  , m_values_ptr(values_ptr)
  , m_first_contiguous(first_contiguous_indexes_policy)
  , m_block(nullptr)
  , m_vblock(block)
  {
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void ItemVectorAccessorT<ValueT>::VectorElement::operator=(const Arccore::Real& value)
  {
    const VectorDistribution& dist = this->m_main_accessor.m_distribution;
    Arccore::ConstArrayView<Arccore::Integer> indices = this->m_entry.getOwnIndexes();
    const Arccore::Integer offset = dist.offset();
    const Arccore::Integer nIndex = indices.size();
    Arccore::ArrayView<Arccore::Real>& values = this->m_values;

    if (m_block) {
      const Arccore::Integer fix_block_size = m_block->size();
      for (Arccore::Integer i = 0; i < nIndex; ++i)
        for (Arccore::Integer j = 0; j < fix_block_size; ++j)
          values[fix_block_size * (indices[i] - offset) + j] = value;
    } else if (m_vblock) {
      for (Arccore::Integer i = 0; i < nIndex; ++i) {
        Arccore::Integer index = indices[i] - offset;
        Arccore::Integer ptr = this->m_values_ptr[index];
        Arccore::Integer block_size = this->m_values_ptr[index + 1] - ptr;
        for (Arccore::Integer j = 0; j < block_size; ++j)
          values[ptr + j] = value;
      }
    } else {
      for (Arccore::Integer i = 0; i < nIndex; ++i)
        values[indices[i] - offset] = value;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /*ALIEN_ACCESSOR_VECTORACCESSORT_H*/
