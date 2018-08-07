/*---------------------------------------------------------------------------*/
/* CoreArray.h                                                 (C) 2000-2018 */
/*                                                                           */
/* Tableau simple pour Arccore.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_COREARRAY_H
#define ARCCORE_BASE_COREARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Tableau simple pour Arccore.
 *
 * Actuellement, ce tableau utilise 'UniqueArray' comme conteneur mais le but
 * est par la suite d'utiliser std::vector.
 *
 * UniqueArray et std::vector ont la même sémantique sauf pour la méthode
 * resize() sur les types de base. Dans le cas de std::vector, si resize() les
 * éléments du tableau compris entre l'ancienne et la nouvelle taille sont
 * initialisés avec le constructeur par défaut (valeurs nulles) alors qu'ils
 * sont inchangés avec UniqueArray.
 */
template<class DataType>
class CoreArray
{
 private:
  typedef std::vector<DataType> ContainerType;
 public:

  //! Type des éléments du tableau
  typedef DataType value_type;
  //! Type de l'itérateur sur un élément du tableau
  typedef typename ContainerType::iterator iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef typename ContainerType::const_iterator const_iterator;
  //! Type pointeur d'un élément du tableau
  typedef typename ContainerType::pointer pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

 public:

  //! Construit un tableau vide.
  CoreArray() {}
  //! Construit un tableau vide.
  CoreArray(ConstArrayView<DataType> v)
  : m_p(v.range().begin(),v.range().end()) {}
 public:

  //! Conversion vers un ConstArrayView
  operator ConstArrayView<DataType>() const
  {
    return CoreArray::_constView(m_p);
  }
  //! Conversion vers un ArrayView
  operator ArrayView<DataType>()
  {
    return CoreArray::_view(m_p);
  }

 public:

  //! i-ème élément du tableau.
  inline DataType& operator[](Integer i)
  {
    return m_p[i];
  }

  //! i-ème élément du tableau.
  inline const DataType& operator[](Integer i) const
  {
    return m_p[i];
  }

  //! Retourne la taille du tableau
  inline Integer size() const { return m_p.size(); }

  //! Retourne un iterateur sur le premier élément du tableau
  inline iterator begin() { return m_p.begin(); }
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  inline iterator end() { return m_p.end(); }
  //! Retourne un iterateur constant sur le premier élément du tableau
  inline const_iterator begin() const { return m_p.begin(); }
  //! Retourne un iterateur constant sur le premier élément après la fin du tableau
  inline const_iterator end() const { return m_p.end(); }

  //! Vue constante
  ConstArrayView<DataType> constView() const
  {
    return CoreArray::_constView(m_p);
  }

  //! Vue modifiable
  ArrayView<DataType> view()
  {
    return CoreArray::_view(m_p);
  }

  //! Retourne \a true si le tableau est vide
  bool empty() const
  {
    return m_p.empty();
  }

  void resize(Integer new_size)
  {
    m_p.resize(new_size);
  }
  void reserve(Integer new_size)
  {
    m_p.reserve(new_size);
  }
  void clear()
  {
    m_p.clear();
  }
  void add(const DataType& v)
  {
    CoreArray::_add(m_p,v);
  }
  DataType& back()
  {
    return m_p.back();
  }
  const DataType& back() const
  {
    return m_p.back();
  }
  const DataType* data() const
  {
    return _data(m_p);
  }
  DataType* data()
  {
    return _data(m_p);
  }
 private:
  static ConstArrayView<DataType> _constView(const std::vector<DataType>& c)
  {
    Integer s = arccoreCheckArraySize(c.size());
    return ConstArrayView<DataType>(s,c.data());
  }
  static ArrayView<DataType> _view(std::vector<DataType>& c)
  {
    Integer s = arccoreCheckArraySize(c.size());
    return ArrayView<DataType>(s,c.data());
  }
  static void _add(std::vector<DataType>& c,const DataType& v)
  {
    c.push_back(v);
  }
  static DataType* _data(std::vector<DataType>& c)
  {
    return c.data();
  }
  static const DataType* _data(const std::vector<DataType>& c)
  {
    return c.data();
  }
 private:

  ContainerType m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
