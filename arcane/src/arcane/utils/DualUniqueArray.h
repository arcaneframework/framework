// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DualUniqueArray.h                                           (C) 2000-2024 */
/*                                                                           */
/* Tableau 1D alloué à la fois sur CPU et accélérateur.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_DUALUNIQUEARRAY_H
#define ARCANE_UTILS_DUALUNIQUEARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arcane/utils/NumArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des DualUniqueArray
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT DualUniqueArrayBase
{
 protected:

  static void _memoryCopy(Span<const std::byte> from, Span<std::byte> to);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Représente un tableau ayant une vue à la fois sur CPU et accélérateur.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de Arcane.
 */
template <typename DataType>
class DualUniqueArray
: public DualUniqueArrayBase
{
  using NumArrayType = NumArray<DataType, MDDim1>;
  using ThatClass = DualUniqueArray<DataType>;

 private:

  class IModifierImpl
  {
   protected:

    virtual ~IModifierImpl() = default;

   public:

    virtual SmallSpan<DataType> view() = 0;
    virtual void resize(Int32 new_size) = 0;
    virtual void endUpdate() = 0;
  };

 public:

  class Modifier
  {
    friend class DualUniqueArray<DataType>;

   private:

    explicit Modifier(IModifierImpl* p)
    : m_p(p)
    {}

   public:

    ~Modifier()
    {
      m_p->endUpdate();
    }

   public:

    SmallSpan<DataType> view() { return m_p->view(); }
    void resize(Int32 new_size) { m_p->resize(new_size); }

   private:

    IModifierImpl* m_p = nullptr;
  };

  class HostModifier
  : public Modifier
  {
    friend class DualUniqueArray<DataType>;

   private:

    explicit HostModifier(ThatClass* data)
    : Modifier(&data->m_array_modifier)
    , m_data(data)
    {}

   public:

    Array<DataType>& container() { return m_data->m_array; }

   private:

    ThatClass* m_data = nullptr;
  };

 private:

  class NumArrayModifierImpl
  : public IModifierImpl
  {
    friend class DualUniqueArray<DataType>;

   private:

    NumArrayModifierImpl(ThatClass* v)
    : m_data(v)
    {}
    SmallSpan<DataType> view() override
    {
      SmallSpan<DataType> v = *(m_data->m_device_array);
      return v;
    }
    void resize(Int32 new_size) override
    {
      m_data->resizeDevice(new_size);
    }
    void endUpdate() override
    {
      m_data->endUpdate(true);
    }

   private:

    ThatClass* m_data = nullptr;
  };

  class UniqueArrayModifierImpl
  : public IModifierImpl
  {
    friend class DualUniqueArray<DataType>;

   public:

    UniqueArrayModifierImpl(ThatClass* v)
    : m_data(v)
    {}
    SmallSpan<DataType> view() override
    {
      return m_data->m_array.view();
    }
    void resize(Int32 new_size) override
    {
      m_data->m_array.resize(new_size);
    }
    void endUpdate() override
    {
      m_data->endUpdate(false);
    }

   private:

    ThatClass* m_data = nullptr;
  };

 public:

  DualUniqueArray()
  : m_numarray_modifier(this)
  , m_array_modifier(this)
  {
  }
  explicit DualUniqueArray(IMemoryAllocator* a)
  : DualUniqueArray()
  {
    m_array = UniqueArray<DataType>(a);
  }

 public:

  SmallSpan<const DataType> hostSmallSpan() const { return m_array.view(); }
  ConstArrayView<DataType> hostView() const { return m_array.view(); }

  void reserve(Int64 capacity)
  {
    m_array.reserve(capacity);
  }
  void resizeHost(Int32 new_size)
  {
    m_array.resize(new_size);
    m_is_valid_numarray = false;
  }
  void fillHost(const DataType& value)
  {
    m_array.fill(value);
    m_is_valid_numarray = false;
  }
  void resizeDevice(Int32 new_size)
  {
    _checkCreateNumArray();
    m_device_array->resize(new_size);
    m_is_valid_array = false;
  }
  void clearHost()
  {
    m_array.clear();
    m_is_valid_numarray = false;
  }
  Int64 size() const { return m_array.size(); }
  SmallSpan<const DataType> view(bool is_device)
  {
    sync(is_device);
    if (is_device) {
      SmallSpan<const DataType> v = *(m_device_array.get());
      return v;
    }
    else {
      return hostSmallSpan();
    }
  }
  void sync(bool is_device)
  {
    if (is_device) {
      _checkUpdateDeviceView();
    }
    else {
      _checkUpdateHostView();
    }
  }
  void endUpdateHost()
  {
    m_is_valid_array = true;
    m_is_valid_numarray = false;
  }
  void endUpdate(bool is_device)
  {
    m_is_valid_array = !is_device;
    m_is_valid_numarray = is_device;
  }
  Modifier modifier(bool is_device)
  {
    if (is_device) {
      _checkCreateNumArray();
      return Modifier(&m_numarray_modifier);
    }
    return Modifier(&m_array_modifier);
  }
  HostModifier hostModifier()
  {
    return HostModifier(this);
  }

 private:

  UniqueArray<DataType> m_array;
  std::unique_ptr<NumArrayType> m_device_array;
  SmallSpan<DataType> m_device_view;
  NumArrayModifierImpl m_numarray_modifier;
  UniqueArrayModifierImpl m_array_modifier;
  bool m_is_valid_array = true;
  bool m_is_valid_numarray = false;

 private:

  void _checkUpdateDeviceView()
  {
    if (!m_is_valid_numarray) {
      _checkCreateNumArray();
      MDSpan<DataType, MDDim1> s(m_array.data(), ArrayIndex<1>(m_array.size()));
      m_device_array->copy(s);
      m_is_valid_numarray = true;
    }
    m_device_view = *(m_device_array.get());
  }
  void _checkUpdateHostView()
  {
    if (!m_is_valid_array) {
      _checkCreateNumArray();
      SmallSpan<const DataType> device_view = *(m_device_array.get());
      m_array.resize(device_view.size());
      _memoryCopy(asBytes(device_view), asWritableBytes(m_array.span()));
      m_is_valid_array = true;
    }
  }
  void _checkCreateNumArray()
  {
    if (!m_device_array) {
      m_device_array = std::make_unique<NumArrayType>(eMemoryRessource::Device);
      if (m_is_valid_array)
        m_device_array->resize(m_array.size());
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
