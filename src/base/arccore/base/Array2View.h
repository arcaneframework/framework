/*---------------------------------------------------------------------------*/
/* Array2View.h                                                (C) 2000-2018 */
/*                                                                           */
/* Vue d'un tableau 2D.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAY2VIEW_H
#define ARCCORE_BASE_ARRAY2VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/base/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vue modifiable pour un tableau 2D.
 *
 * Comme toute vue, une instance de cette classe n'est valide que tant
 * que le conteneur dont elle est issue ne change pas de nombre d'éléments.
 */
template<class DataType>
class Array2View
{
 public:
  friend class ConstArray2View<DataType>;
 public:
  //! Créé une vue 2D de dimension [\a dim1_size][\a dim2_size]
  Array2View(DataType* ptr,Integer dim1_size,Integer dim2_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size)
  {
  }
  //! Créé une vue 2D vide.
  Array2View() : m_ptr(0), m_dim1_size(0), m_dim2_size(0)
  {
  }
 public:
  //! Nombre d'éléments de la première dimension
  Integer dim1Size() const { return m_dim1_size; }
  //! Nombre d'éléments de la deuxième dimension
  Integer dim2Size() const { return m_dim2_size; }
  //! Nombre total d'éléments.
  Integer totalNbElement() const { return m_dim1_size*m_dim2_size; }
 public:
  ArrayView<DataType> operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ArrayView<DataType>(m_dim2_size,m_ptr + (m_dim2_size*i));
  }
  ConstArrayView<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ConstArrayView<DataType>(m_dim2_size,m_ptr + (m_dim2_size*i));
  }
  //! Valeur de l'élément [\a i][\a j]
  DataType item(Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    ARCCORE_CHECK_AT(j,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
  //! Positionne l'élément [\a i][\a j] à \a value
  DataType setItem(Integer i,Integer j,const DataType& value)
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    ARCCORE_CHECK_AT(j,m_dim2_size);
    m_ptr[(m_dim2_size*i) + j] = value;
  }
 public:
  /*!
   * \brief Pointeur sur la mémoire allouée.
   */
  inline DataType* unguardedBasePointer()
  { return m_ptr; }
  /*!
   * \brief Pointeur sur la mémoire allouée.
   */
  inline DataType* data() { return m_ptr; }
 private:
  DataType* m_ptr;
  Integer m_dim1_size;
  Integer m_dim2_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue pour un tableau 2D constant.
 */
template<class DataType>
class ConstArray2View
{
 public:
  ConstArray2View(const DataType* ptr,Integer dim1_size,Integer dim2_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size)
  {
  }
  ConstArray2View(const Array2View<DataType>& rhs)
  : m_ptr(rhs.m_ptr), m_dim1_size(rhs.m_dim1_size), m_dim2_size(rhs.m_dim2_size)
  {
  }
  ConstArray2View() : m_ptr(0), m_dim1_size(0), m_dim2_size(0)
  {
  }
 public:
  Integer dim1Size() const { return m_dim1_size; }
  Integer dim2Size() const { return m_dim2_size; }
  Integer totalNbElement() const { return m_dim1_size*m_dim2_size; }
 public:
  ConstArrayView<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ConstArrayView<DataType>(m_dim2_size,m_ptr + (m_dim2_size*i));
  }
  DataType item(Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    ARCCORE_CHECK_AT(j,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  } 
 public:
  /*!
   * \brief Pointeur sur la mémoire allouée.
   */
  inline const DataType* unguardedBasePointer() const
  { return m_ptr; }
  /*!
   * \brief Pointeur sur la mémoire allouée.
   */
  inline const DataType* data() const { return m_ptr; }
 private:
  const DataType* m_ptr;
  Integer m_dim1_size;
  Integer m_dim2_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
