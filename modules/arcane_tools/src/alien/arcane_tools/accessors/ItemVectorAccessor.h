#ifndef ALIEN_ACCESSOR_ITEMVECTORACCESSOR_H
#define ALIEN_ACCESSOR_ITEMVECTORACCESSOR_H

#include <arcane/IItemFamily.h>
#include <arcane/anyitem/AnyItem.h>

#include "alien/arcane_tools/IIndexManager.h"
#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/data/IVector.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/handlers/scalar/BaseVectorWriter.h>
#include <arcane/Item.h>
#include <arcane/utils/Array.h>
#include <arcane/utils/AutoRef.h>
#include <arcane/utils/ObjectImpl.h>

#include <arcane/IVariable.h>
#include <arcane/MeshVariableArrayRef.h>
#include "alien/AlienArcaneToolsPrecomp.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename T> class SimpleCSRVector;

  template <typename VarT, typename ValueT> class ScalarVariableT;
  template <typename VarT, typename ValueT> class ScalarArrayVariableT;
  template <typename VarT, typename ValueT> class ArrayVariableT;
  template <typename VarT, typename ValueT> class ScalarAnyItemVariableT;
  template <typename VarT, typename ValueT> class ScalarArrayAnyItemVariableT;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  class ItemVectorAccessorT : public Common::VectorWriterBaseT<ValueT>
  {
   public:
    typedef ValueT ValueType;

    //! enum to define the sub block extracting policy
    typedef enum {
      FirstContiguousIndexes,
      LastContiguousIndexes,
      UserSpecified,
      Undefined
    } eSubBlockExtractingPolicyType;

   public:
    class VectorElement
    {
     public:
      struct Generic
      {
        struct OpSet
        {
          template <typename DataT> static void apply(DataT& dest, const DataT& src)
          {
            dest = src;
          }
        };
        struct OpAdd
        {
          template <typename DataT> static void apply(DataT& dest, const DataT& src)
          {
            dest += src;
          }
        };
        struct OpSub
        {
          template <typename DataT> static void apply(DataT& dest, const DataT& src)
          {
            dest -= src;
          }
        };
        template <typename Op, typename VarT>
        static void process(VectorElement& v, const VarT& var);
        template <typename Op, typename VarT>
        static void process(VarT& var, const VectorElement& v);
        template <typename Op, typename VarT>
        static void process(
            VectorElement& v, const ScalarAnyItemVariableT<VarT, ValueT>& var);
        template <typename Op, typename VarT>
        static void process(
            ScalarAnyItemVariableT<VarT, ValueT>& var, const VectorElement& v);
        template <typename Op, typename VarT>
        static void process(
            VectorElement& v, const ScalarArrayAnyItemVariableT<VarT, ValueT>& var);
        template <typename Op, typename VarT>
        static void process(
            ScalarArrayAnyItemVariableT<VarT, ValueT>& var, const VectorElement& v);
        template <typename Op, typename VarT>
        static void process(
            VectorElement& v, const VarT& var, Arccore::Integer block_size);
        template <typename Op, typename VarT>
        static void process(
            VarT& var, const VectorElement& v, Arccore::Integer block_size);
      };

     public:
      VectorElement(const VectorElement&) = default;

      VectorElement(ItemVectorAccessorT& accessor, const Block* block,
          const IIndexManager::Entry& entry, Arccore::ArrayView<ValueT> values);

      VectorElement(ItemVectorAccessorT& accessor, const VBlock* block,
          const IIndexManager::Entry& entry, Arccore::ArrayView<ValueT> values,
          Arccore::ConstArrayView<Arccore::Integer> values_ptr,
          bool isFirstContiguousIndexes);

      virtual ~VectorElement() {}

      inline bool isVariableBlockSize() const
      {
        if (m_vblock)
          return true;
        else
          return false;
      }

      inline Arccore::Integer getBlockSize() const
      {
        if (m_block)
          return m_block->size();
        else if (m_vblock)
          return m_vblock->maxBlockSize();
        else
          throw Arccore::FatalErrorException(A_FUNCINFO, "No block infos");
      }

     public:
      template <typename VarT>
      inline void operator=(const ScalarVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSet>(*this, var);
      }
      template <typename VarT>
      inline void operator=(const ScalarArrayVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSet>(*this, var);
      }
      template <typename VarT>
      inline void operator=(const ArrayVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSet>(
            *this, var, this->getBlockSize());
      }
      template <typename VarT>
      inline void operator=(const ScalarAnyItemVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSet>(*this, var);
      }
      template <typename VarT>
      inline void operator=(const ScalarArrayAnyItemVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSet>(*this, var);
      }

      template <typename VarT>
      inline void operator+=(const ScalarVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpAdd>(*this, var);
      }
      template <typename VarT>
      inline void operator+=(const ScalarArrayVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpAdd>(*this, var);
      }
      template <typename VarT>
      inline void operator+=(const ArrayVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpAdd>(
            *this, var, this->getBlockSize());
      }
      template <typename VarT>
      inline void operator+=(const ScalarAnyItemVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpAdd>(*this, var);
      }
      template <typename VarT>
      inline void operator+=(const ScalarArrayAnyItemVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpAdd>(*this, var);
      }

      template <typename VarT>
      inline void operator-=(const ScalarVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSub>(*this, var);
      }
      template <typename VarT>
      inline void operator-=(const ScalarArrayVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSub>(*this, var);
      }
      template <typename VarT>
      inline void operator-=(const ArrayVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSub>(
            *this, var, this->getBlockSize());
      }
      template <typename VarT>
      inline void operator-=(const ScalarAnyItemVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSub>(*this, var);
      }
      template <typename VarT>
      inline void operator-=(const ScalarArrayAnyItemVariableT<VarT, ValueT>& var)
      {
        Generic::template process<typename Generic::OpSub>(*this, var);
      }

      void operator=(const Arccore::Real& value);

     private:
      const IIndexManager::Entry& m_entry;
      ItemVectorAccessorT& m_main_accessor;
      Arccore::ArrayView<ValueT> m_values;
      Arccore::ConstArrayView<Arccore::Integer> m_values_ptr;
      bool m_first_contiguous;
      const Block* m_block;
      const VBlock* m_vblock;
    };

    ItemVectorAccessorT(IVector& vector, bool update = true);

    virtual ~ItemVectorAccessorT() {}

    VectorElement operator()(const IIndexManager::Entry& entry,
        eSubBlockExtractingPolicyType opt = FirstContiguousIndexes);

   private:
    const ISpace& m_space;
    const VectorDistribution& m_distribution;
    const Block* m_block;
    const VBlock* m_vblock;
    Arccore::ConstArrayView<Arccore::Integer> m_values_ptr;
  };

  /*---------------------------------------------------------------------------*/

  template <typename VarT, typename ValueT = Arccore::Real> class ScalarVariableT
  {
   private:
    typedef typename ItemVectorAccessorT<ValueT>::VectorElement::Generic Generic;

   public:
    ScalarVariableT(VarT& var)
    : m_var(var)
    , m_item_internals(
          const_cast<Arcane::IItemFamily*>(var.variable()->itemFamily())->itemsInternal())
    {
    }

    inline void operator=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSet>(*this, v);
    }
    inline void operator+=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpAdd>(*this, v);
    }
    inline void operator-=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSub>(*this, v);
    }

    inline bool checkKind(const Arccore::Integer kind) const
    {
      return m_var.variable()->itemKind() == kind;
    }
    inline Arccore::Real& operator[](const Arcane::Item& item) { return m_var[item]; }
    inline const Arccore::Real& operator[](const Arcane::Item& item) const
    {
      return m_var[item];
    }
    inline Arccore::Real& valueAt(const Arccore::Integer localId)
    {
      return m_var[m_item_internals[localId]];
    }
    inline const Arccore::Real& valueAt(const Arccore::Integer localId) const
    {
      return m_var[m_item_internals[localId]];
    }

   private:
    VarT& m_var;
    const Arcane::ItemInternalArrayView m_item_internals;
  };

  template <typename VarT, typename ValueT = Arccore::Real> class ScalarArrayVariableT
  {
   private:
    typedef typename ItemVectorAccessorT<ValueT>::VectorElement::Generic Generic;

   public:
    ScalarArrayVariableT(VarT& var, const Arccore::Integer index)
    : m_var(var)
    , m_index(index)
    , m_item_internals(
          const_cast<Arcane::IItemFamily*>(var.variable()->itemFamily())->itemsInternal())
    {
    }

    inline void operator=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSet>(*this, v);
    }
    inline void operator+=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpAdd>(*this, v);
    }
    inline void operator-=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSub>(*this, v);
    }

    inline bool checkKind(const Arccore::Integer kind) const
    {
      return m_var.variable()->itemKind() == kind;
    }
    inline Arccore::Real& operator[](const Arcane::Item& item)
    {
      return m_var[item][m_index];
    }
    inline const Arccore::Real& operator[](const Arcane::Item& item) const
    {
      return m_var[item][m_index];
    }
    inline Arccore::Real& valueAt(const Arccore::Integer localId)
    {
      return m_var[m_item_internals[localId]][m_index];
    }
    inline const Arccore::Real& valueAt(const Arccore::Integer localId) const
    {
      return m_var[m_item_internals[localId]][m_index];
    }

   private:
    VarT& m_var;
    const Arccore::Integer m_index;
    const Arcane::ItemInternalArrayView m_item_internals;
  };

  template <typename VarT, typename ValueT = Arccore::Real> class ArrayVariableT
  {
   private:
    typedef typename ItemVectorAccessorT<ValueT>::VectorElement::Generic Generic;
#if (ARCANE_VERSION < 11801)
    typedef typename VarT::ArrayType VarTArrayType;
    typedef typename VarT::ConstArrayType VarTConstArrayType;
#else /* ARCANE_VERSION */
    typedef typename VarT::ReturnReferenceType VarTArrayType;
    typedef typename VarT::ConstReturnReferenceType VarTConstArrayType;
#endif /* ARCANE_VERSION */

   public:
    ArrayVariableT(VarT& var)
    : m_var(var)
    , m_item_internals(
          const_cast<Arcane::IItemFamily*>(var.variable()->itemFamily())->itemsInternal())
    {
    }

    ScalarArrayVariableT<VarT, ValueT> operator[](const Arccore::Integer i) const
    {
      return ScalarArrayVariableT<VarT, ValueT>(m_var, i);
    }

    inline void operator=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSet>(*this, v, v.getBlockSize());
    }
    inline void operator+=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpAdd>(*this, v, v.getBlockSize());
    }
    inline void operator-=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSub>(*this, v, v.getBlockSize());
    }

    inline bool checkKind(const Arccore::Integer kind) const
    {
      return m_var.variable()->itemKind() == kind;
    }
    inline VarTArrayType operator[](const Arcane::Item& item) { return m_var[item]; }
    inline VarTConstArrayType operator[](const Arcane::Item& item) const
    {
      return m_var[item];
    }
    inline VarTArrayType valueAt(const Arccore::Integer localId)
    {
      return m_var[m_item_internals[localId]];
    }
    inline VarTConstArrayType valueAt(const Arccore::Integer localId) const
    {
      return m_var[m_item_internals[localId]];
    }

   private:
    VarT& m_var;
    const Arcane::ItemInternalArrayView m_item_internals;
  };

  template <typename VarT, typename ValueT = Arccore::Real> class ScalarAnyItemVariableT
  {
   private:
    typedef typename ItemVectorAccessorT<ValueT>::VectorElement::Generic Generic;
    friend struct ItemVectorAccessorT<ValueT>::VectorElement::Generic; // pour l'acc�s �
                                                                       // m_var par
                                                                       // process : a
                                                                       // remplacer par
                                                                       // des acces

   public:
    ScalarAnyItemVariableT(VarT& var)
    : m_var(var)
    {
    }

    inline void operator=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSet>(*this, v);
    }
    inline void operator+=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpAdd>(*this, v);
    }
    inline void operator-=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSub>(*this, v);
    }

   private:
    VarT& m_var;
  };

  template <typename VarT, typename ValueT = Arccore::Real>
  class ScalarArrayAnyItemVariableT
  {
   private:
    typedef typename ItemVectorAccessorT<ValueT>::VectorElement::Generic Generic;
    friend class ItemVectorAccessorT<ValueT>::VectorElement::Generic; // pour l'acc�s �
    // m_var par process

   public:
    ScalarArrayAnyItemVariableT(VarT& var, const Arccore::Integer index)
    : m_var(var)
    , m_index(index)
    {
    }

    inline void operator=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSet>(*this, v);
    }
    inline void operator+=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpAdd>(*this, v);
    }
    inline void operator-=(const typename ItemVectorAccessorT<ValueT>::VectorElement& v)
    {
      Generic::template process<typename Generic::OpSub>(*this, v);
    }

   private:
    VarT& m_var;
    const Arccore::Integer m_index;
  };

  template <typename VarT, typename ValueT = Arccore::Real> class ArrayAnyItemVariableT
  {
   private:
    typedef typename ItemVectorAccessorT<ValueT>::VectorElement::Generic Generic;

    //#if (ARCANE_VERSION < 11801)
    //  typedef typename VarT::ArrayType VarTArrayType;
    //  typedef typename VarT::ConstArrayType VarTConstArrayType;
    //#else /* ARCANE_VERSION */
    //  typedef typename VarT::ReturnReferenceType VarTArrayType;
    //  typedef typename VarT::ConstReturnReferenceType VarTConstArrayType;
    //#endif /* ARCANE_VERSION */

   public:
    ArrayAnyItemVariableT(VarT& var)
    : m_var(var)
    {
    }
    ScalarArrayAnyItemVariableT<VarT, ValueT> operator[](const Arccore::Integer i) const
    {
      return ScalarArrayAnyItemVariableT<VarT, ValueT>(m_var, i);
    }

    // NOT YET IMPLEMENTED :  only used to map to ScalarArrayAnyItemVariableT
    // inline void operator=(const typename ItemVectorAccessorT<ValueT>::VectorElement &
    // v) { Generic::template process<typename Generic::OpSet>(*this,v,v.getBlockSize());
    // }
    // inline void operator+=(const typename ItemVectorAccessorT<ValueT>::VectorElement &
    // v) { Generic::template process<typename Generic::OpAdd>(*this,v,v.getBlockSize());
    // }
    // inline void operator-=(const typename ItemVectorAccessorT<ValueT>::VectorElement &
    // v) { Generic::template process<typename Generic::OpSub>(*this,v,v.getBlockSize());
    // }

    // inline bool checkKind(const Integer kind) const { return
    // m_var.variable()->itemKind() == kind;  }
    // inline VarTArrayType operator[](const Item & item) { return m_var[item]; }
    // inline VarTConstArrayType operator[](const Item & item) const { return m_var[item];
    // }
    // inline VarTArrayType valueAt(const Integer localId) { return
    // m_var[m_item_internals[localId]]; }
    // inline VarTConstArrayType valueAt(const Integer localId) const { return
    // m_var[m_item_internals[localId]]; }

   private:
    VarT& m_var;
  };

  /*---------------------------------------------------------------------------*/

  template <typename DataT>
  ScalarVariableT<Arcane::ItemVariableScalarRefT<DataT>> Variable(
      Arcane::ItemVariableScalarRefT<DataT>& var)
  {
    return ScalarVariableT<Arcane::ItemVariableScalarRefT<DataT>>(var);
  }

  template <typename DataT>
  ScalarVariableT<Arcane::ItemPartialVariableScalarRefT<DataT>> Variable(
      Arcane::ItemPartialVariableScalarRefT<DataT>& var)
  {
    return ScalarVariableT<Arcane::ItemPartialVariableScalarRefT<DataT>>(var);
  }

  template <typename DataT>
  ArrayVariableT<Arcane::ItemVariableArrayRefT<DataT>> Variable(
      Arcane::ItemVariableArrayRefT<DataT>& var)
  {
    return ArrayVariableT<Arcane::ItemVariableArrayRefT<DataT>>(var);
  }

  template <typename DataT>
  ArrayVariableT<Arcane::ItemPartialVariableArrayRefT<DataT>> Variable(
      Arcane::ItemPartialVariableArrayRefT<DataT>& var)
  {
    return ArrayVariableT<Arcane::ItemPartialVariableArrayRefT<DataT>>(var);
  }

  template <typename DataT>
  ScalarAnyItemVariableT<Arcane::AnyItem::Variable<DataT>> Variable(
      Arcane::AnyItem::Variable<DataT>& var)
  {
    return ScalarAnyItemVariableT<Arcane::AnyItem::Variable<DataT>>(var);
  }

  template <typename DataT>
  ArrayAnyItemVariableT<Arcane::AnyItem::VariableArray<DataT>> Variable(
      Arcane::AnyItem::VariableArray<DataT>& var)
  {
    return ArrayAnyItemVariableT<Arcane::AnyItem::VariableArray<DataT>>(var);
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      typename ItemVectorAccessorT<ValueT>::VectorElement& v, const VarT& var)
  {
    // ALIEN_ASSERT((var.checkKind(v.m_entry.getKind())),("Bad kind variable operation"));
    Arccore::ConstArrayView<Arccore::Integer> indices = v.m_entry.getOwnIndexes();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();
    const Arccore::Integer nIndex = indices.size();
    Arccore::ConstArrayView<Arccore::Integer> localIds = v.m_entry.getOwnLocalIds();
    Arccore::ArrayView<Arccore::Real>& values = v.m_values;
    for (Arccore::Integer i = 0; i < nIndex; ++i)
      Op::apply(values[indices[i] - offset], var.valueAt(localIds[i]));
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      VarT& var, typename ItemVectorAccessorT<ValueT>::VectorElement const& v)
  {
    // ALIEN_ASSERT((var.checkKind(v.m_entry.getKind())),("Bad kind variable operation"));
    Arccore::ConstArrayView<Arccore::Integer> indices = v.m_entry.getOwnIndexes();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();
    const Arccore::Integer nIndex = indices.size();
    Arccore::ConstArrayView<Arccore::Integer> localIds = v.m_entry.getOwnLocalIds();
    const Arccore::ArrayView<Arccore::Real>& values = v.m_values;
    for (Arccore::Integer i = 0; i < nIndex; ++i)
      Op::apply(var.valueAt(localIds[i]), values[indices[i] - offset]);
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      typename ItemVectorAccessorT<ValueT>::VectorElement& v, const VarT& var,
      Arccore::Integer block_size)
  {
    // ALIEN_ASSERT((var.checkKind(v.m_entry.getKind())),("Bad kind variable operation"));
    Arccore::ConstArrayView<Arccore::Integer> indices = v.m_entry.getOwnIndexes();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();
    const Arccore::Integer nIndex = indices.size();
    Arccore::ConstArrayView<Arccore::Integer> localIds = v.m_entry.getOwnLocalIds();
    Arccore::ArrayView<ValueT>& values = v.m_values;
    if (v.isVariableBlockSize()) {
      Arccore::ConstArrayView<Arccore::Integer> values_ptr = v.m_values_ptr;
      for (Arccore::Integer i = 0; i < nIndex; ++i) {
        Arccore::Integer lid = indices[i] - offset;
        Arccore::Integer ptr = values_ptr[lid];
        Arccore::Integer nuk = values_ptr[lid + 1] - values_ptr[lid];
        Arccore::Integer offset = (v.m_first_contiguous ? 0 : block_size - nuk);
        for (Arccore::Integer j = 0; j < nuk; ++j)
          Op::apply(values[ptr + j], var.valueAt(localIds[i])[offset + j]);
      }
    } else {
      for (Arccore::Integer i = 0; i < nIndex; ++i)
        for (Arccore::Integer j = 0; j < block_size; ++j)
          Op::apply(values[block_size * (indices[i] - offset) + j],
              var.valueAt(localIds[i])[j]);
    }
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(VarT& var,
      typename ItemVectorAccessorT<ValueT>::VectorElement const& v,
      Arccore::Integer block_size)
  {
    // ALIEN_ASSERT((var.checkKind(v.m_entry.getKind())),("Bad kind variable operation"));
    Arccore::ConstArrayView<Arccore::Integer> indices = v.m_entry.getOwnIndexes();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();
    const Arccore::Integer nIndex = indices.size();
    Arccore::ConstArrayView<Arccore::Integer> localIds = v.m_entry.getOwnLocalIds();
    const Arccore::ArrayView<ValueT>& values = v.m_values;
    if (v.isVariableBlockSize()) {
      Arccore::ConstArrayView<Arccore::Integer> values_ptr = v.m_values_ptr;
      for (Arccore::Integer i = 0; i < nIndex; ++i) {
        Arccore::Integer lid = indices[i] - offset;
        Arccore::Integer ptr = values_ptr[lid];
        Arccore::Integer nuk = values_ptr[lid + 1] - values_ptr[lid];
        Arccore::Integer offset = (v.m_first_contiguous ? 0 : block_size - nuk);
        for (Arccore::Integer j = 0; j < nuk; ++j) {
          Op::apply(var.valueAt(localIds[i])[offset + j], values[ptr + j]);
        }
      }
    } else {
      for (Arccore::Integer i = 0; i < nIndex; ++i)
        for (Arccore::Integer j = 0; j < block_size; ++j)
          Op::apply(var.valueAt(localIds[i])[j],
              values[block_size * (indices[i] - offset) + j]);
    }
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      typename ItemVectorAccessorT<ValueT>::VectorElement& v,
      const ScalarAnyItemVariableT<VarT, ValueT>& var)
  {
#ifndef HAVE_TUPLE_IFPEN /* own local ids ordonn�s croissant */
    // Actuellement BasicIndexManager ne trie pas (cf
    // BasicIndexManager::MyEntryImpl::finalize)
    Arccore::ConstArrayView<Arccore::Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    Arccore::ConstArrayView<Arccore::Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Arccore::Integer nIndex = indices.size();
// ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
// localids"));
#else // HAVE_TUPLE_IFPEN /* own local ids non ordonn�s */
    ConstArrayView<Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    ConstArrayView<Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Integer nIndex = indices.size();
    // ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
    // localids"));

    // Tri sym�trique ordonn� sur les local_ids pour les regrouper par AnyItem::Group (les
    // indices suivent le r�ordonnancement)
    typedef ConstArrayView<Integer>::iterator iterator;
    DualRandomIterator<iterator, iterator> begin(local_ids.begin(), indices.begin());
    DualRandomIterator<iterator, iterator> end = begin + nIndex;
    FirstIndexComparator<DualRandomIterator<iterator, iterator>> comparator;
    std::sort(begin, end, comparator);
#endif // HAVE_TUPLE_IFPEN

    // const Integer offset = v.m_main_accessor.m_space.indexManager()->minLocalIndex();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();

    Arccore::ArrayView<Arccore::Real>& values =
        v.m_values; // donn�es du vecteur alg�brique

    const Arcane::AnyItem::Variable<ValueT>& any_var = var.m_var;
    const Arcane::AnyItem::Family& family = any_var.family();
    const Arccore::Integer group_size = family.groupSize();

    Arccore::Integer i = 0;
    Arccore::Integer first_group_index = 0;
    for (Arccore::Integer igrp = 0; ((igrp < group_size) && (i < nIndex)); ++igrp) {
      Arcane::ItemGroup group = family.group(igrp);
      const Arccore::Integer last_group_index = first_group_index + group.size();

      const Arcane::IVariable* ivar = any_var.variables()[igrp];
      if (ivar == NULL) { // AnyItem::Variable peut ne pas �tre compl�tement d�fini
        // Il faut avancer la zone de transfert au del� du groupe courant
        while (i < nIndex and local_ids[i] < last_group_index)
          ++i;
      } else if (local_ids[i] < last_group_index) { // Il y a du boulot... (sinon aucune
                                                    // donn�e associ�e � ce groupe, local
                                                    // ids trop loin)
        // ALIEN_ASSERT((ivar->isPartial() == family.isPartial(group)),("Inconsistent
        // partial variable indicator"));

        Arccore::ConstArrayView<ValueT> var_values = any_var.valuesAtGroup(igrp);
        if (ivar->isPartial()) {
          // ALIEN_ASSERT((ivar->itemGroup() == group),("Inconsistent variable group"));
          // Donn�es de variable partielle index�es par les positions dans le groupe
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(values[indices[i] - offset],
                var_values[local_ids[i] - first_group_index]);
            ++i;
          }
        } else {
          // Variable non partielle index�e par les item local ids
          Arccore::Int32ConstArrayView item_local_ids = group.view().localIds();
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(values[indices[i] - offset],
                var_values[item_local_ids[local_ids[i] - first_group_index]]);
            ++i;
          }
        }
      }
      first_group_index = last_group_index;
    }
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      ScalarAnyItemVariableT<VarT, ValueT>& var,
      typename ItemVectorAccessorT<ValueT>::VectorElement const& v)
  {
#ifndef HAVE_TUPLE_IFPEN /* own local ids ordonn�s croissant */
    // Actuellement BasicIndexManager ne trie pas (cf
    // BasicIndexManager::MyEntryImpl::finalize)
    Arccore::ConstArrayView<Arccore::Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    Arccore::ConstArrayView<Arccore::Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Arccore::Integer nIndex = indices.size();
// ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
// localids"));
#else // HAVE_TUPLE_IFPEN /* own local ids non ordonn�s */
    ConstArrayView<Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    ConstArrayView<Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Integer nIndex = indices.size();
    // ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
    // localids"));

    // Tri sym�trique ordonn� sur les local_ids pour les regrouper par AnyItem::Group (les
    // indices suivent le r�ordonnancement)
    typedef ConstArrayView<Integer>::iterator iterator;
    DualRandomIterator<iterator, iterator> begin(local_ids.begin(), indices.begin());
    DualRandomIterator<iterator, iterator> end = begin + nIndex;
    FirstIndexComparator<DualRandomIterator<iterator, iterator>> comparator;
    std::sort(begin, end, comparator);
#endif // HAVE_TUPLE_IFPEN

    // const Integer offset = v.m_main_accessor.m_space.indexManager()->minLocalIndex();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();

    const Arccore::ArrayView<Arccore::Real>& values =
        v.m_values; // donn�es du vecteur alg�brique

    Arcane::AnyItem::Variable<ValueT>& any_var = var.m_var;
    const Arcane::AnyItem::Family& family = any_var.family();
    const Arccore::Integer group_size = family.groupSize();

    Arccore::Integer i = 0;
    Arccore::Integer first_group_index = 0;
    for (Arccore::Integer igrp = 0; igrp < group_size and i < nIndex; ++igrp) {
      Arcane::ItemGroup group = family.group(igrp);
      const Arccore::Integer last_group_index = first_group_index + group.size();

      const Arcane::IVariable* ivar = any_var.variables()[igrp];
      if (ivar == NULL) { // AnyItem::Variable peut ne pas �tre compl�tement d�fini
        // Il faut avancer la zone de transfert au del� du groupe courant
        while (i < nIndex and local_ids[i] < last_group_index)
          ++i;
      } else if (local_ids[i] < last_group_index) { // Il y a du boulot... (sinon aucune
                                                    // donn�e associ�e � ce groupe, local
                                                    // ids trop loin)
        // ALIEN_ASSERT((ivar->isPartial() == family.isPartial(group)),("Inconsistent
        // partial variable indicator"));

        Arccore::ArrayView<ValueT> var_values = any_var.valuesAtGroup(igrp);
        if (ivar->isPartial()) {
          // ALIEN_ASSERT((ivar->itemGroup() == group),("Inconsistent variable group"));
          // Donn�es de variable partielle index�es par les positions dans le groupe
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(var_values[local_ids[i] - first_group_index],
                values[indices[i] - offset]);
            ++i;
          }
        } else {
          // Variable non partielle index�e par les item local ids
          Arccore::Int32ConstArrayView item_local_ids = group.view().localIds();
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(var_values[item_local_ids[local_ids[i] - first_group_index]],
                values[indices[i] - offset]);
            ++i;
          }
        }
      }
      first_group_index = last_group_index;
    }
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      typename ItemVectorAccessorT<ValueT>::VectorElement& v,
      const ScalarArrayAnyItemVariableT<VarT, ValueT>& var)
  {
#ifndef HAVE_TUPLE_IFPEN /* own local ids ordonn�s croissant */
    // Actuellement BasicIndexManager ne trie pas (cf
    // BasicIndexManager::MyEntryImpl::finalize)
    Arccore::ConstArrayView<Arccore::Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    Arccore::ConstArrayView<Arccore::Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Arccore::Integer nIndex = indices.size();
// ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
// localids"));
#else // HAVE_TUPLE_IFPEN /* own local ids non ordonn�s */
    ConstArrayView<Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    ConstArrayView<Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Integer nIndex = indices.size();
    // ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
    // localids"));

    // Tri sym�trique ordonn� sur les local_ids pour les regrouper par AnyItem::Group (les
    // indices suivent le r�ordonnancement)
    typedef ConstArrayView<Integer>::iterator iterator;
    DualRandomIterator<iterator, iterator> begin(local_ids.begin(), indices.begin());
    DualRandomIterator<iterator, iterator> end = begin + nIndex;
    FirstIndexComparator<DualRandomIterator<iterator, iterator>> comparator;
    std::sort(begin, end, comparator);
#endif // HAVE_TUPLE_IFPEN

    // const Integer offset = v.m_main_accessor.m_space.indexManager()->minLocalIndex();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();

    Arccore::ArrayView<Arccore::Real>& values =
        v.m_values; // donn�es du vecteur alg�brique

    const Arcane::AnyItem::Variable<ValueT>& any_var = var.m_var;
    const Arccore::Integer scalar_array_index = var.m_index;
    const Arcane::AnyItem::Family& family = any_var.family();
    const Arccore::Integer group_size = family.groupSize();

    Arccore::Integer i = 0;
    Arccore::Integer first_group_index = 0;
    for (Arccore::Integer igrp = 0; ((igrp < group_size) && (i < nIndex)); ++igrp) {
      Arcane::ItemGroup group = family.group(igrp);
      const Arccore::Integer last_group_index = first_group_index + group.size();

      const Arcane::IVariable* ivar = any_var.variables()[igrp];
      if (ivar == NULL) { // AnyItem::Variable peut ne pas �tre compl�tement d�fini
        // Il faut avancer la zone de transfert au del� du groupe courant
        while (i < nIndex and local_ids[i] < last_group_index)
          ++i;
      } else if (local_ids[i] < last_group_index) { // Il y a du boulot... (sinon aucune
                                                    // donn�e associ�e � ce groupe, local
                                                    // ids trop loin)
        // ALIEN_ASSERT((ivar->isPartial() == family.isPartial(group)),("Inconsistent
        // partial variable indicator"));

        Arccore::ConstArray2View<ValueT> var_values = any_var.valuesAtGroup(igrp);
        if (ivar->isPartial()) {
          // ALIEN_ASSERT((ivar->itemGroup() == group),("Inconsistent variable group"));
          // Donn�es de variable partielle index�es par les positions dans le groupe
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(values[indices[i] - offset],
                var_values[local_ids[i] - first_group_index][scalar_array_index]);
            ++i;
          }
        } else {
          // Variable non partielle index�e par les item local ids
          Arccore::Int32ConstArrayView item_local_ids = group.view().localIds();
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(values[indices[i] - offset],
                var_values[item_local_ids[local_ids[i] - first_group_index]]
                          [scalar_array_index]);
            ++i;
          }
        }
      }
      first_group_index = last_group_index;
    }
  }

  template <typename ValueT>
  template <typename Op, typename VarT>
  void ItemVectorAccessorT<ValueT>::VectorElement::Generic::process(
      ScalarArrayAnyItemVariableT<VarT, ValueT>& var,
      typename ItemVectorAccessorT<ValueT>::VectorElement const& v)
  {
#ifndef HAVE_TUPLE_IFPEN /* own local ids ordonn�s croissant */
    // Actuellement BasicIndexManager ne trie pas (cf
    // BasicIndexManager::MyEntryImpl::finalize)
    Arccore::ConstArrayView<Arccore::Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    Arccore::ConstArrayView<Arccore::Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Arccore::Integer nIndex = indices.size();
// ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
// localids"));
#else // HAVE_TUPLE_IFPEN /* own local ids non ordonn�s */
    ConstArrayView<Integer> indices =
        v.m_entry.getOwnIndexes(); // Indexation dans le vecteur alg�brique
    ConstArrayView<Integer> local_ids =
        v.m_entry.getOwnLocalIds(); // localIds au sens de AnyItem::Family
    const Integer nIndex = indices.size();
    // ALIEN_ASSERT((local_ids.size() == indices.size()),("Incompatible entry indexes /
    // localids"));

    // Tri sym�trique ordonn� sur les local_ids pour les regrouper par AnyItem::Group (les
    // indices suivent le r�ordonnancement)
    typedef ConstArrayView<Integer>::iterator iterator;
    DualRandomIterator<iterator, iterator> begin(local_ids.begin(), indices.begin());
    DualRandomIterator<iterator, iterator> end = begin + nIndex;
    FirstIndexComparator<DualRandomIterator<iterator, iterator>> comparator;
    std::sort(begin, end, comparator);
#endif // HAVE_TUPLE_IFPEN

    // const Integer offset = v.m_main_accessor.m_space.indexManager()->minLocalIndex();
    const Arccore::Integer offset = v.m_main_accessor.m_distribution.offset();

    const Arccore::ArrayView<Arccore::Real>& values =
        v.m_values; // donn�es du vecteur alg�brique

    Arcane::AnyItem::VariableArray<ValueT>& any_var = var.m_var;
    const Arccore::Integer scalar_array_index = var.m_index;
    const Arcane::AnyItem::Family& family = any_var.family();
    const Arccore::Integer group_size = family.groupSize();

    Arccore::Integer i = 0;
    Arccore::Integer first_group_index = 0;
    for (Arccore::Integer igrp = 0; igrp < group_size and i < nIndex; ++igrp) {
      Arcane::ItemGroup group = family.group(igrp);
      const Arccore::Integer last_group_index = first_group_index + group.size();

      const Arcane::IVariable* ivar = any_var.variables()[igrp];
      if (ivar == NULL) { // AnyItem::Variable peut ne pas �tre compl�tement d�fini
        // Il faut avancer la zone de transfert au del� du groupe courant
        while (i < nIndex and local_ids[i] < last_group_index)
          ++i;
      } else if (local_ids[i] < last_group_index) { // Il y a du boulot... (sinon aucune
                                                    // donn�e associ�e � ce groupe, local
                                                    // ids trop loin)
        // ALIEN_ASSERT((ivar->isPartial() == family.isPartial(group)),("Inconsistent
        // partial variable indicator"));

        Arccore::Array2View<ValueT> var_values = any_var.valuesAtGroup(igrp);
        if (ivar->isPartial()) {
          // ALIEN_ASSERT((ivar->itemGroup() == group),("Inconsistent variable group"));
          // Donn�es de variable partielle index�es par les positions dans le groupe
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(var_values[local_ids[i] - first_group_index][scalar_array_index],
                values[indices[i] - offset]);
            ++i;
          }
        } else {
          // Variable non partielle index�e par les item local ids
          Arccore::Int32ConstArrayView item_local_ids = group.view().localIds();
          while (i < nIndex and local_ids[i] < last_group_index) {
            Op::apply(var_values[item_local_ids[local_ids[i] - first_group_index]]
                                [scalar_array_index],
                values[indices[i] - offset]);
            ++i;
          }
        }
      }
      first_group_index = last_group_index;
    }
  }

  typedef ItemVectorAccessorT<Arccore::Real> ItemVectorAccessor;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Si on est en release et que l'on souhaite l'inlining, VectorAccessorT est inclus ici
#ifndef ALIEN_INCLUDE_TEMPLATE_IN_CC
#include <alien/arcane_tools/accessors/ItemVectorAccessorT.h>
#endif /* ALIEN_INCLUDE_TEMPLATE_IN_CC */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_ACCESSOR_VECTORACCESSOR_H */
