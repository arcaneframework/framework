# Coding Rules {#arcanedoc_general_codingrules}

[TOC]

To ensure that the different modules developed for the ARCANE platform have a
certain homogeneity, this document proposes a set of coding rules.

## General {#arcanedoc_general_codingrules_general}

- the language used is C++20
- file encoding must be 'UTF-8' with BOM at the beginning of the file. The first
  line of each file must be as follows:
  // -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
- all comments use the **Doxygen** product syntax to be able to extract paper or
  hypertext documentation directly from the source code,
- all identifiers are written in **English**.
- indentations are done with 2 spaces. There must be no tab characters in the
  code.

## Source Files {#arcanedoc_general_codingrules_source}

- all source files are formatted the same way and all begin with a header
  describing the file name, the modification date.
- the actual code follows. Each identifier must be commented and functions must
  be separated by two lines of comments. For example:

```cpp
/*!
 * \brief an example function.
 *
 * This is the long description of the example function.
 *
 * \param argc number of arguments
 * \param argv array of argument values
 * \return the number of arguments divided by 2
 */
int
functionExample(int argc,char** argv)
{
  return argc/2;
}
```


## Variables {#arcanedoc_general_codingrules_variable}

- variable names are always in **lowercase**. If the name is composed of several
  logical words, each word is separated by the underscore character. For
  example:
  - \c volume
  - \c list_of_element
  - \c cells
- To avoid any ambiguity, plural names are reserved for container type
  variables; all other names must be in the **singular**.

## Classes {#arcanedoc_general_codingrules_classe}

- Class names (\c class) start with a capital letter and continue with lowercase
  letters. If the name is composed of several logical words, the first letter of
  each new word is capitalized. For example:
  - \c Component
  - \c ComponentMng
  - \c String

- Class members, in addition to respecting the same conventions as any variable,
  will always be prefixed by the two characters <tt>m_</tt>. For example:
  - \c m_volume
  - \c m_list_of_element

## Methods and Functions {#arcanedoc_general_codingrules_method}

In the following, the word function will be used to designate both functions and
class methods.

- function names are always in <b>lowercase</b>. If the name is composed of
  several logical words, the first letter of each new word is capitalized. For
  example:
  - \c numberOfElement()
  - \c assign()
- if the method name corresponds to the notion of a property (i.e., semantically
  equivalent to a class field) named \a value, the accessor must be the property
  name (\a value()) and the method to change the value must be \a setValue().
  <strong>The accessor must not start with `get`</strong>. If the
  property is boolean, it is possible to prefix the accessor with \a is. For
  example \a isEmpty().

- To avoid any ambiguity, all names must be in the <b>singular</b>
- Function definitions must be on at least two lines:
  - the first includes the return type and possibly the class name if it is a
    method.
  - the second necessarily includes the function name.
  - followed by the list of arguments on the second line and subsequent lines.
- The opening brace of the function body and the closing brace must be on a
  separate line:
```cpp
int
function1(int argc,char** argv)
{
  return argc/2;
}
```


## Example {#arcanedoc_general_codingrules_example}

```cpp
/*!
 * \brief Constant array of type \a T.
 * 
 * This class encapsulates a standard constant C array (pointer) and its number
 * of elements. Access to its elements is done via the operator operator[]().
 * The base() method allows obtaining the array pointer to pass it
 * to standard C functions.
 * 
 * The instance only keeps a pointer to the beginning of the C array and performs
 * no memory management. The developer must ensure that the pointer
 * remains valid as long as the instance exists.
 * 
 * The elements of the array cannot be modified.
 * 
 * In debug mode, an overflow check is performed when accessing
 * the operator operator[]().
 */
template<typename T>
class ConstCArrayT
{
 private:

 protected:

 public:
	
  //! Type of the array elements
  typedef T value_type;
  //! Type of the constant iterator on an array element
  typedef const value_type * key_restrict const_iterator;
  //! Type constant pointer of an array element
  typedef const value_type * key_restrict const_pointer;
  //! Type constant reference of an array element
  typedef const value_type& const_reference;
  //! Type indexing the array
  typedef Integer size_type;
  //! Type of a distance between iterator elements of the array
  typedef ptrdiff_t difference_type;

  //! Type of a constant iterator on the entire array
  typedef ConstIterT< ConstCArrayT<T> > const_iter;

 public:

  //! Constructs an empty array.
  ConstCArrayT() : m_size(0), m_ptr(0) {}
  //! Constructs an array with \a s elements
  explicit ConstCArrayT(Integer s,const T* ptr)
  : m_size(s), m_ptr(ptr) {}
  /*! \brief Copy constructor.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  ConstCArrayT(const ConstCArrayT<T>& from)
  : m_size(from.m_size), m_ptr(from.m_ptr) {}
  /*! \brief Copy constructor.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  ConstCArrayT(const CArrayBaseT<T>& from)
  : m_size(from.size()), m_ptr(from.begin())
    {
    }

  /*! \brief Copy assignment operator.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  const ConstCArrayT<T>& operator=(const ConstCArrayT<T>& from)
    { m_size=from.m_size; m_ptr=from.m_ptr; return *this; }
	
  /*! \brief Copy assignment operator.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  const ConstCArrayT<T>& operator=(const CArrayBaseT<T>& from)
    {
      m_size = from.size();
      m_ptr  = from.begin();
      return (*this);
    }

 public:

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, overflows are checked.
   */
  const T& operator[](Integer i) const
    {
      return m_ptr[i];
    }

  //! Number of elements in the array
  inline Integer size() const { return m_size; }
  //! Iterator on the first element of the array
  inline const_iterator begin() const { return m_ptr; }
  //! Iterator on the first element after the end of the array
  inline const_iterator end() const { return m_ptr+m_size; }
  //! \a true if the array is empty (size()==0)
  inline bool empty() const { return m_size==0; }

  //! Pointer to the start of the array.
  inline const T* base() const { return m_ptr; }

 protected:

 private:

  Integer m_size; //!< Number of elements 
  const T* m_ptr; //!< Pointer to the start of the array
};
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_general
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_getting_started_basicstruct
</span> -->
</div>
