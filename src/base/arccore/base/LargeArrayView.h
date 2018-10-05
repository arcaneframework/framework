/*---------------------------------------------------------------------------*/
/* LargeArrayView.h                                            (C) 2000-2018 */
/*                                                                           */
/* Types définissant les vues de tableaux C dont la taille est un Int64.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_LARGEARRAYVIEW_H
#define ARCCORE_BASE_LARGEARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue modifiable d'un tableau d'un type \a T.
 *
 Cette classe template permet d'accéder et d'utiliser un tableau d'éléments du
 type \a T de la même manière qu'un tableau C standard. Elle est similaire à
 ArrayView à ceci près que le nombre d'éléments est stocké sur un 'Int64' et
 peut donc dépasser 2Go.
*/
template<class T>
class LargeArrayView
{
 public:

  //! Type des éléments du tableau
  typedef T value_type;
  //! Type pointeur d'un élément du tableau
  typedef value_type* pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type de l'itérateur sur un élément du tableau
  typedef ArrayIterator<pointer> iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Int64 size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef std::ptrdiff_t difference_type;

  //! Type d'un itérateur sur tout le tableau
  typedef IterT< ArrayView<T> > iter;
  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT< ArrayView<T> > const_iter;

 public:


 public:

  //! Construit une vue vide.
  LargeArrayView() : m_ptr(nullptr), m_size(0) {}
  //! Constructeur de recopie depuis une autre vue
  LargeArrayView(const ArrayView<T>& from)
  : m_ptr(from.m_ptr), m_size(from.m_size) {}
  LargeArrayView(const LargeArrayView<T>& from)
  : m_ptr(from.m_ptr), m_size(from.m_size) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  LargeArrayView(Int64 asize,T* ptr)
  : m_ptr(ptr), m_size(asize) {}
  //! Opérateur de recopie
  const LargeArrayView<T>& operator=(const LargeArrayView<T>& from)
  {
    m_ptr = from.m_ptr;
    m_size = from.m_size;
    return *this;
  }

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline T& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& item(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Positionne le i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline void setItem(Int64 i,const T& v)
  {
    ARCCORE_CHECK_AT(i,m_size);
    m_ptr[i] = v;
  }

  //! Retourne la taille du tableau
  inline Int64 size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Int64 length() const { return m_size; }

  /*!
   * \brief Itérateur sur le premier élément du tableau.
   */
  iterator begin() { return iterator(m_ptr); }
  /*!
   * \brief Itérateur sur le premier élément après la fin du tableau.
   */
  iterator end() { return iterator(m_ptr+m_size); }
  /*!
   * \brief Itérateur constant sur le premier élément du tableau.
   */
  const_iterator begin() const { return iterator(m_ptr); }
  /*!
   * \brief Itérateur constant sur le premier élément après la fin du tableau.
   */
  const_iterator end() const { return iterator(m_ptr+m_size); }

 public:

  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_ptr,m_ptr+m_size);
  }
  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_size);
  }

 public:
  //! Addresse du index-ème élément
  inline T* ptrAt(Int64 index)
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  //! Addresse du index-ème élément
  inline const T* ptrAt(Int64 index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  // Elément d'indice \a i. Vérifie toujours les débordements
  const T& at(Int64 i) const
  {
    arccoreCheckAt(i,m_size);
    return m_ptr[i];
  }

  // Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Int64 i,const T& value)
  {
    arccoreCheckAt(i,m_size);
    m_ptr[i] = value;
  }

  //! Remplit le tableau avec la valeur \a o
  inline void fill(T o)
  {
    for( Int64 i=0, n=m_size; i<n; ++i )
      m_ptr[i] = o;
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  ConstArrayView<T> constView() const
  {
    return ConstArrayView<T>(m_size,m_ptr);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si \a (\a abegin+ \a asize) est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  LargeArrayView<T> subView(Int64 abegin,Int64 asize)
  {
    if (abegin>=m_size)
      return ArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ArrayView<T>(asize,m_ptr+abegin);
  }

  /*!
   * \brief Sous-vue constante à partir de
   * l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin+ \a asize) est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Int64 abegin,Int64 asize) const
  {
    if (abegin>=m_size)
      return ConstArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ConstArrayView<T>(asize,m_ptr+abegin);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  LargeArrayView<T> subViewInterval(Int64 index,Int64 nb_interval)
  {
    Int64 n = m_size;
    Int64 isize = n / nb_interval;
    Int64 ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index+1)==nb_interval)
      isize = n - ibegin;
    return LargeArrayView<T>(isize,m_ptr+ibegin);
  }

  /*!\brief Recopie le tableau \a copy_array dans l'instance.
   * Comme aucune allocation mémoire n'est effectuée, le
   * nombre d'éléments de \a copy_array doit être inférieur ou égal au
   * nombre d'éléments courant. S'il est inférieur, les éléments du
   * tableau courant situés à la fin du tableau sont inchangés
   */
  template<class U>
  inline void copy(const U& copy_array)
  {
    ARCCORE_ASSERT( (copy_array.size()<=m_size), ("Bad size %d %d",copy_array.size(),m_size) );
    const T* copy_begin = copy_array.unguardedBasePointer();
    T* to_ptr = m_ptr;
    Int64 n = copy_array.size();
    for( Int64 i=0; i<n; ++i )
      to_ptr[i] = copy_begin[i];
  }

  //! Retourne \a true si le tableau est vide (dimension nulle)
  bool empty() const { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const T& v) const
  {
    for( Int64 i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }

 public:

  void setArray(const ArrayView<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }
  void setArray(const LargeArrayView<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointeur sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  const_pointer data() const
  { return m_ptr; }

  /*!
   * \brief Pointeur constant sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  pointer data()
  { return m_ptr; }

 protected:
  
  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setArray(T* v,Int64 s){ m_ptr = v; m_size = s; }

  /*!
   * \brief Modifie le pointeur du début du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setPtr(T* v) { m_ptr = v; }

  /*!
   * \brief Modifie la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setSize(Int64 s) { m_size = s; }

 private:

  T* m_ptr;  //!< Pointeur sur le tableau
  Int64 m_size; //!< Nombre d'éléments du tableau

 private:

  static Int64 _min(Int64 a,Int64 b)
  {
    return ( (a<b) ? a : b );
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection 
 * \brief Vue constante d'un tableau de type \a T.
 *
 * Cette classe fonctionne de la même manière que LargeArrayView à la seule
 * différence qu'il n'est pas possible de modifier les éléments du tableau.
 */
template<class T>
class ConstLargeArrayView
{
 public:

  //! Type des éléments du tableau
  typedef T value_type;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Int64 size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef std::ptrdiff_t difference_type;

 public:

  //! Construit un tableau vide.
  ConstLargeArrayView() : m_ptr(nullptr), m_size(0) {}
  //! Construit un tableau avec \a s élément
  ConstLargeArrayView(Int64 s,const T* ptr)
  : m_ptr(ptr), m_size(s) {}
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstLargeArrayView(const ConstLargeArrayView<T>& from)
  : m_ptr(from.m_ptr), m_size(from.m_size) {}
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstLargeArrayView(const LargeArrayView<T>& from)
  : m_ptr(from.data()), m_size(from.size()) { }
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstLargeArrayView(const ConstArrayView<T>& from)
  : m_ptr(from.data()), m_size(from.size()) { }
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstLargeArrayView(const ArrayView<T>& from)
  : m_ptr(from.data()), m_size(from.size()) { }

  /*!
   * \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstLargeArrayView<T>& operator=(const ConstLargeArrayView<T>& from)
  {
    m_ptr=from.m_ptr;
    m_size=from.m_size;
    return *this;
  }

  /*! \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstLargeArrayView<T>& operator=(const LargeArrayView<T>& from)
  {
    m_ptr  = from.data();
    m_size = from.size();
    return (*this);
  }

  /*! \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstLargeArrayView<T>& operator=(const ConstArrayView<T>& from)
  {
    m_ptr  = from.data();
    m_size = from.size();
    return (*this);
  }

  /*! \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstLargeArrayView<T>& operator=(const ArrayView<T>& from)
  {
    m_ptr  = from.data();
    m_size = from.size();
    return (*this);
  }

 public:

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   contenant \a asize éléments.
   *
   * Si \a (abegin+asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstLargeArrayView<T> subView(Int64 abegin,Int64 asize) const
  {
    if (abegin>=m_size)
      return ConstLargeArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ConstLargeArrayView<T>(asize,m_ptr+abegin);
  }

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   * contenant \a asize éléments.
   *
   * Si \a (abegin+asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstLargeArrayView<T> subConstView(Int64 abegin,Int64 asize) const
  {
    return subView(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  ArrayView<T> subViewInterval(Int64 index,Int64 nb_interval)
  {
    Int64 n = m_size;
    Int64 isize = n / nb_interval;
    Int64 ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index+1)==nb_interval)
      isize = n - ibegin;
    ARCCORE_CHECK_AT(ibegin+isize,n);
    return ConstLargeArrayView<T>(isize,m_ptr+ibegin);
  }

  //! Addresse du index-ème élément
  inline const T* ptrAt(Int64 index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  const T& operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& item(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  //! Nombre d'éléments du tableau
  inline Int64 size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Int64 length() const { return m_size; }
  //! Itérateur sur le premier élément du tableau.
  const_iterator begin() const { return const_iterator(m_ptr); }
  //! Itérateur sur le premier élément après la fin du tableau.
  const_iterator end() const { return const_iterator(m_ptr+m_size); }
  //! \a true si le tableau est vide (size()==0)
  inline bool empty() const { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const T& v) const
  {
    for( Int64 i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }
  void setArray(const ConstArrayView<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }
  void setArray(const ConstLargeArrayView<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  const T* data() const  { return m_ptr; }
  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_size);
  }

 private:

  const T* m_ptr; //!< Pointeur sur le début du tableau
  Int64 m_size; //!< Nombre d'éléments 

 private:

  static Int64 _min(Int64 a,Int64 b)
  {
    return ( (a<b) ? a : b );
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(ConstLargeArrayView<T> rhs, ConstLargeArrayView<T> lhs)
{
  if (rhs.size()!=lhs.size())
    return false;
  Int64 s = rhs.size();
  for( Int64 i=0; i<s; ++i ){
    if (rhs[i]==lhs[i])
      continue;
    else
      return false;
  }
  return true;
}

template<typename T> inline bool
operator!=(ConstLargeArrayView<T> rhs, ConstLargeArrayView<T> lhs)
{
  return !(rhs==lhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(LargeArrayView<T> rhs, LargeArrayView<T> lhs)
{
  return operator==(rhs.constView(),lhs.constView());
}

template<typename T> inline bool
operator!=(LargeArrayView<T> rhs, LargeArrayView<T> lhs)
{
  return !(rhs==lhs);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Affiche sur le flot \a o les valeurs du tableau \a val.
 *
 * Si \a max_print est positif, alors au plus \a max_print valeurs
 * sont affichées. Si la taille du tableau est supérieure à
 * \a max_print, alors les (max_print/2) premiers et derniers
 * éléments sont affichés.
 */
template<typename T> inline void
dumpArray(std::ostream& o,ConstLargeArrayView<T> val,int max_print)
{
  Int64 n = val.size();
  if (max_print>0 && n>max_print){
    // N'affiche que les (max_print/2) premiers et les (max_print/2) dernièrs
    // sinon si le tableau est très grand cela peut générer des
    // sorties listings énormes.
    Int64 z = (max_print/2);
    Int64 z2 = n - z;
    o << "[0]=\"" << val[0] << '"';
    for( Int64 i=1; i<z; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
    o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
    for( Int64 i=(z2+1); i<n; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
  }
  else{
    for( Int64 i=0; i<n; ++i ){
      if (i!=0)
        o << ' ';
      o << "[" << i << "]=\"" << val[i] << '"';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, ConstLargeArrayView<T> val)
{
  dumpArray(o,val,500);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, LargeArrayView<T> val)
{
  o << val.constView();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
