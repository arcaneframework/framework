// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Utils.h                                         (C) 2000-2025             */
/*                                                                           */
/* Neo utils                                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NEO_UTILS_H
#define NEO_UTILS_H

#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <string>
#include <chrono>
#include <iterator>
#include <fstream>
#include <memory>

#ifdef NDEBUG
static constexpr bool ndebug = true;
static constexpr bool _debug = false;
#ifdef _MSC_VER
#define _MS_REL_
#endif
#else
static constexpr bool ndebug = false;
static constexpr bool _debug = true;
#endif

//----------------------------------------------------------------------------/
//----------------------------------------------------------------------------/

namespace Neo
{

enum class Trace{
  Verbose, Silent, VerboseInFile
};

struct NullBuffer : public std::streambuf
{
  int overflow(int c) override {
    return c;
  }
};

struct NeoOutputStream
{
  explicit NeoOutputStream(Neo::Trace trace_level)
  : m_null_ostream{&m_null_buffer}
  , m_trace_level(trace_level) {
    _openOutputFile();
  }

  explicit NeoOutputStream(Neo::Trace trace_level, int rank)
  : m_null_ostream{&m_null_buffer}
  , m_trace_level(trace_level)
  , m_rank(rank) {
    _openOutputFile();
  }

  ~NeoOutputStream() {
    if (m_file_stream && m_file_stream->is_open())
      m_file_stream->close();
  }

  mutable std::ostream m_null_ostream;
  Neo::Trace m_trace_level;
  int m_rank = 0;
  std::string m_file_name;
  std::shared_ptr<std::ofstream> m_file_stream = nullptr;
  NullBuffer m_null_buffer;

  std::string const& fileName() const {return m_file_name;}

  std::ostream& stream() const {
    switch (m_trace_level) {
    case Neo::Trace::Silent:
      return m_null_ostream;
      break;
    case Neo::Trace::Verbose:
      return std::cout;
      break;
    case Neo::Trace::VerboseInFile:
      return *m_file_stream;
      break;
    }
    return m_null_ostream;
  }

private:
  void _openOutputFile() {
    if (m_trace_level == Trace::VerboseInFile) {
      m_file_name = std::string{"Neo_output_"} + std::to_string(m_rank) + ".txt";
      m_file_stream = std::make_unique<std::ofstream>(m_file_name,std::ios::out | std::ios::app);
      if (!m_file_stream->is_open())
        throw std::runtime_error(std::string{"Cannot open file"} + m_file_name);
    }
  }
};

/*!
 *
 * @param rank: current process/subdomain number (0 by default)
 * @return output stream
 *
 * Neo debug logs are activated by environment variables:
 * - NEO_DEBUG_PRINT=1 to activate console debug outputs
 * - NEO_DEBUG_PRINT_IN_FILE=1 to activate file debug outputs (file named Neo_output_rank.txt)
 */
inline NeoOutputStream print(int rank = 0) {
  if (std::getenv("NEO_DEBUG_PRINT"))
    return NeoOutputStream{Neo::Trace::Verbose};
  else if (std::getenv("NEO_DEBUG_PRINT_IN_FILE"))
    return NeoOutputStream{Neo::Trace::VerboseInFile, rank};
  else
    return NeoOutputStream{Neo::Trace::Silent};
}

//----------------------------------------------------------------------------/

namespace utils
{
  using Int64 = std::int64_t;
  using Int32 = std::int32_t;
  using Real = double;
  static constexpr utils::Int32 NULL_ITEM_LID = -1;

  //----------------------------------------------------------------------------/

  struct Real3
  {
    double x, y, z;
    // Utilities
    friend inline std::ostream& operator<<(std::ostream& oss, Neo::utils::Real3 const& real3) {
      oss << "{" << real3.x << "," << real3.y << "," << real3.z << "}";
      return oss;
    }
    friend inline bool operator==(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b) {
      return a.x == b.x && a.y == b.y && a.z == b.z;
    }
    friend inline bool operator!=(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b) {
      return !(a == b);
    }
    friend inline Real3 operator+(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b) {
      return Real3{ a.x + b.x, a.y + b.y, a.z + b.z };
    }
    friend inline Real3 operator-(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b) {
      return Real3{ a.x - b.x, a.y - b.y, a.z - b.z };
    }
    friend inline Real3 operator*(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b) {
      return Real3{ a.x * b.x, a.y * b.y, a.z * b.z };
    }
    friend inline Real3 operator/(Neo::utils::Real3 const& a, Neo::utils::Real3 const& b) {
      return Real3{ a.x / b.x, a.y / b.y, a.z / b.z };
    }
    friend inline Real3& operator+=(Neo::utils::Real3& a, Neo::utils::Real3 const& b) {
      a = a + b;
      return a;
    }
    friend inline Real3& operator-=(Neo::utils::Real3& a, Neo::utils::Real3 const& b) {
      a = a - b;
      return a;
    }
    friend inline Real3& operator*=(Neo::utils::Real3& a, Neo::utils::Real3 const& b) {
      a = a * b;
      return a;
    }
    friend inline Real3& operator/=(Neo::utils::Real3& a, Neo::utils::Real3 const& b) {
      a = a / b;
      return a;
    }
  };

  //----------------------------------------------------------------------------/

  /*!
     * \brief View of a data array
     * @tparam T view data type
     * Will be replaced by std::span when moving to C++20
    */
  // todo use std::span instead when moving to C++20
  template <typename T>
  struct Span
  {
    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using size_type = int;
    using vector_size_type = typename std::vector<non_const_value_type>::size_type;

    T* m_ptr = nullptr;
    size_type m_size = 0;

    Span(T* data, size_type size)
    : m_size(size)
    , m_ptr(data) {}

    Span(T* data, vector_size_type size)
    : m_size(size)
    , m_ptr(data) {}

    Span() = default;

    T& operator[](int i) {
      assert(i < m_size);
      return *(m_ptr + i);
    }

    T const& operator[](int i) const {
      assert(i < m_size);
      return *(m_ptr + i);
    }

    T* begin() { return m_ptr; }
    T* end() { return m_ptr + m_size; }

    int size() const { return m_size; }

    std::vector<non_const_value_type> copy() {
      std::vector<non_const_value_type> vec(m_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  //----------------------------------------------------------------------------/

  template <typename T>
  struct ConstSpan
  {
    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using size_type = int;
    using vector_size_type = typename std::vector<non_const_value_type>::size_type;

    const T* m_ptr = nullptr;
    int m_size = 0;

    ConstSpan(const T* data, size_type size)
    : m_size(size)
    , m_ptr(data) {}

    ConstSpan(const T* data, vector_size_type size)
    : m_size(size)
    , m_ptr(data) {}

    ConstSpan() = default;

    const T& operator[](int i) const {
      assert(i < m_size);
      return *(m_ptr + i);
    }

    const T* begin() const { return m_ptr; }
    const T* end() const { return m_ptr + m_size; }

    int size() const { return m_size; }

    std::vector<non_const_value_type> copy() {
      std::vector<non_const_value_type> vec(m_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  //----------------------------------------------------------------------------/

  /*!
   * \brief 2-Dimensional view of a contiguous data chunk
   * The second dimension varies first {(i,j),(i,j+1),(i+1,j),(i+1,j+1)}...
   * \fn operator[i] returns a view of size \refitem Array2View.m_dim2_size
   * @tparam T view data type
   * Will be replaced by std::mdspan when moving to C++23
   */
  // todo use std::mdspan instead when moving to C++23
  template <typename T>
  struct Span2
  {
    using value_type = T;
    using size_type = int;

    T* m_ptr = nullptr;
    size_type m_dim1_size = 0;
    size_type m_dim2_size = 0;

    Span<T> operator[](int i) {
      assert(i < m_dim1_size);
      return { m_ptr + i * m_dim2_size, m_dim2_size };
    }

    T* begin() {return m_ptr;}
    T* end() { return m_ptr + (m_dim1_size * m_dim2_size); }

    size_type dim1Size() const { return m_dim1_size; }
    size_type dim2Size() const { return m_dim2_size; }

    std::vector<T> copy() {
      std::vector<T> vec(m_dim1_size * m_dim2_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  //----------------------------------------------------------------------------/

  /*!
   * 2-Dimensional const view. cf. \refitem Array2View
   * @tparam T view data type
   */
  template <typename T>
  struct ConstSpan2
  {
    using value_type = T;
    using size_type = int;

    T* m_ptr = nullptr;
    size_type m_dim1_size = 0;
    size_type m_dim2_size = 0;

    ConstSpan<T> operator[](int i) const {
      assert(i < m_dim1_size);
      return { m_ptr + i * m_dim2_size, m_dim2_size };
    }

    const T* begin() {return m_ptr;}
    const T* end()   {return m_ptr+(m_dim1_size*m_dim2_size);}

    size_type dim1Size() const { return m_dim1_size; }
    size_type dim2Size() const { return m_dim2_size; }

    std::vector<T> copy() { std::vector<T> vec(m_dim1_size*m_dim2_size);
      std::copy(this->begin(), this->end(), vec.begin());
      return vec;
    }
  };

  //----------------------------------------------------------------------------/

  template <typename Container>
  std::ostream&  _printContainer(Container&& container, std::ostream& oss){
    std::copy(container.begin(), container.end(), std::ostream_iterator<typename std::remove_reference_t<Container>::value_type>(oss, " "));
    oss << std::endl;
    return oss;
  }



  //----------------------------------------------------------------------------/

  using namespace std::chrono_literals;

  struct Profiler
  {

    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end;
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    void stop() { m_end = std::chrono::high_resolution_clock::now(); }
    void start(std::string name) {
      m_name = name;
      start();
    }
    void stop_and_print(std::string message) {
      stop();
      print(message);
    }
    void stop_and_print() {
      stop();
      print("");
    }
    void print(std::string message) {
      std::cout << "=== cpu time " << message << m_name << " = " << (m_end - m_start) / 1us << " microseconds ===" << std::endl;
    }
  };

  //----------------------------------------------------------------------------/

  // Some useful using
  using Int32Span = Span<Int32>;
  using Int64Span = Span<Int64>;
  using Int32ConstSpan = ConstSpan<Int32>;
  using Int64ConstSpan = ConstSpan<Int64>;
  using Int32Span2D = Span2<Int32>; // todo check name see stl MDSpan et Arcane ?
  using Int64Span2D = Span2<Int64>;
  using Int32ConstSpan2D = ConstSpan2<Int32>; // todo check name see stl MDSpan et Arcane ?
  using Int64SConstpan2D = ConstSpan2<Int64>;

} // end namespace utils

//----------------------------------------------------------------------------/

} // end namespace Neo

//----------------------------------------------------------------------------/

// Array utilities

#ifdef _MSC_VER // problem with operator<< overload with MSVC
namespace std
{ // is undefined behavior, but can't find another way with MSVC. MSVC cannot find an operator<< outside the arguments' namespace...
#endif

template <typename T>
std::ostream& operator<<(std::ostream& oss, std::vector<T> const& container) {
  return Neo::utils::_printContainer(container, oss);
}

template <typename T>
std::ostream& operator<<(std::ostream& oss, Neo::utils::Span<T> const& container) {
  return Neo::utils::_printContainer(container, oss);
}

template <typename T>
std::ostream& operator<<(std::ostream& oss, Neo::utils::ConstSpan<T> const& container) {
  return Neo::utils::_printContainer(container, oss);
}

#ifdef _MSC_VER // problem with operator<< overload with MSVC
} // namespace std
#endif

#ifdef _MSC_VER // problem with operator<< overload with MSVC
namespace Neo
{ // MSVC cannot find an operator<< outside the arguments' namespace...
#endif

template <typename T>
std::ostream& operator<<(Neo::NeoOutputStream const& oss, T const& printable) {
  return oss.stream() << printable;
}

#ifdef _MSC_VER // problem with operator<< overload with MSVC
} // namespace Neo
#endif

namespace Neo
{
  namespace utils
  {
    template <typename Container>
    void printContainer(Container&& container, std::string const& name = "Container", int rank = 0) {
      std::ostringstream oss;
      oss << name << ", size : " << container.size() << std::endl;
      _printContainer(container, oss);
      Neo::print(rank) << oss.str();
    }
    template <typename Container>
    void printContainer(std::ostringstream& oss, Container&& container, std::string const& name = "Container") {
      oss << name << ", size : " << container.size() << std::endl;
      _printContainer(container, oss);
    }
  }
}



//----------------------------------------------------------------------------/
//----------------------------------------------------------------------------/

#endif // NEO_UTILS_H
