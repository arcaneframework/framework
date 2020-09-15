//
// Created by dechaiss on 5/15/20.
//

#ifndef NEO_UTILS_H
#define NEO_UTILS_H

/*-------------------------
 * sdc - (C) 2020
 * NEtwork Oriented kernel
 * Utils
 *--------------------------
 */

#include <iostream>
#include <vector>
#include <cassert>

namespace Neo {
namespace utils {
  using Int64 = long int;
  using Int32 = int;
  struct Real3 { double x,y,z;};
  template <typename T>
  struct ArrayView {
    using value_type = T;
    std::size_t m_size = 0;
    T* m_ptr = nullptr;
    T& operator[](int i) {assert(i<m_size); return *(m_ptr+i);}
    T* begin() {return m_ptr;}
    T* end()   {return m_ptr+m_size;}
    std::size_t size() const {return m_size;}
    std::vector<T> copy() { std::vector<T> vec(m_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  template <typename T>
  struct ConstArrayView {
    using value_type = T;
    std::size_t m_size = 0;
    const T* m_ptr = nullptr;
    const T& operator[](int i) const {assert(i<m_size); return *(m_ptr+i);}
    const T* begin() const {return m_ptr;}
    const T* end() const  {return m_ptr+m_size;}
    std::size_t size() const {return m_size;}
    std::vector<T> copy() { std::vector<T> vec(m_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  /*!
   * \brief 2-Dimensional view of a contiguous data chunk
   * The second dimension varies first {(i,j),(i,j+1),(i+1,j),(i+1,j+1)}...
   * \fn operator[i] returns a view of size \refitem Array2View.m_dim2_size
   * @tparam T
   */
  template <typename T>
  struct Array2View {
    using value_type = T;
    int m_dim1_size = 0;
    int m_dim2_size = 0;
    T* m_ptr = nullptr;
    ArrayView<T> operator[](int i) {assert(i<m_dim1_size); return {(std::size_t)m_dim2_size,m_ptr+i*m_dim2_size};}
    T* begin() {return m_ptr;}
    T* end()   {return m_ptr+(m_dim1_size*m_dim2_size);}
    int dim1Size() const { return m_dim1_size; }
    int dim2Size() const { return m_dim2_size; }
    std::vector<T> copy() { std::vector<T> vec(m_dim1_size*m_dim2_size);
      std::copy(this->begin(),this->end(), vec.begin());
      return vec;
    }
  };

  /*!
   * 2-Dimensional const view. cf. \refitem Array2View
   * @tparam T
   */
  template <typename T>
  struct ConstArray2View {
    using value_type = T;
    int m_dim1_size = 0;
    int m_dim2_size = 0;
    T* m_ptr = nullptr;
    ConstArrayView<T> operator[](int i) const {assert(i<m_dim1_size); return {(std::size_t)m_dim2_size,m_ptr+i*m_dim2_size};}
    const T* begin() {return m_ptr;}
    const T* end()   {return m_ptr+(m_dim1_size*m_dim2_size);}
    int dim1Size() const { return m_dim1_size; }
    int dim2Size() const { return m_dim2_size; }
    std::vector<T> copy() { std::vector<T> vec(m_dim1_size*m_dim2_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  static constexpr utils::Int32 NULL_ITEM_LID = -1;

  template <typename Container>
  std::ostream&  _printContainer(Container&& container, std::ostream& oss){
    std::copy(container.begin(),container.end(),std::ostream_iterator<typename std::remove_reference_t<Container>::value_type>(oss," "));
    return oss;
  }

  template <typename Container>
  void printContainer(Container&& container, std::string const& name="Container"){
    std::cout << name << " , size : " << container.size() << std::endl;
    _printContainer(container, std::cout);
    std::cout << std::endl;
  }

}// end namespace utils

}// end namespace Neo

// Real3 utilities
inline
std::ostream& operator<<(std::ostream& oss, Neo::utils::Real3 const& real3){
  oss << "{" << real3.x  << ","  << real3.y << "," << real3.z << "}";
  return oss;
}
inline
bool operator==(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b){
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Array utilities
template <typename T>
std::ostream& operator<<(std::ostream& oss, std::vector<T> const& container)
{
  return Neo::utils::_printContainer(container, oss);
}

template <typename T>
std::ostream& operator<<(std::ostream& oss, Neo::utils::ArrayView<T> const& container)
{
  return Neo::utils::_printContainer(container, oss);
}

template <typename T>
std::ostream& operator<<(std::ostream& oss, Neo::utils::ConstArrayView<T> const& container)
{
  return Neo::utils::_printContainer(container, oss);
}



#endif // NEO_UTILS_H
