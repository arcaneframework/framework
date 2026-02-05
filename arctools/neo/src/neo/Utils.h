// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Utils.h                                         (C) 2000-2026             */
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

static std::ostream& nullStream() {
  static NullBuffer nullBuffer;
  static std::ostream nullStream(&nullBuffer);
  return nullStream;
}

struct NeoOutputStream
{
  explicit NeoOutputStream(Neo::Trace trace_level, int rank = 0)
    : m_trace_level(trace_level)
    , m_rank(rank) {
      switch (m_trace_level) {
      case Trace::Verbose:
        m_stream = &std::cout;
        break;
      case Trace::Silent:
        m_stream = &nullStream();
        break;
      case Trace::VerboseInFile:
        m_file_name = std::string{"Neo_output_"} + std::to_string(m_rank) + ".txt";
        m_file_stream = std::make_unique<std::ofstream>(m_file_name,std::ios::out | std::ios::app);
        if (!m_file_stream->is_open())
          throw std::runtime_error(std::string{"Cannot open file"} + m_file_name);
        m_stream = m_file_stream.get();
      }
  }

  ~NeoOutputStream() {
    if (! m_stream) return;
    try {
      m_stream->flush();
    }
    catch (...) {}
  }

  std::ostream* m_stream = nullptr;
  Neo::Trace m_trace_level;
  int m_rank = 0;
  std::string m_file_name;
  std::shared_ptr<std::ofstream> m_file_stream = nullptr;
  static NullBuffer m_null_buffer;

  std::string const& fileName() const {return m_file_name;}

  std::ostream& stream() {
    return *m_stream;
  }
};

inline NeoOutputStream& endline(NeoOutputStream& oss) {
  oss.stream() << std::endl;
  return oss;
};

using NeoOutputStreamHandler = NeoOutputStream& (*)(NeoOutputStream&);

class Printer
{
  NeoOutputStream m_oss;
public:
  NeoOutputStream& operator()() {
    return m_oss;
  }
  explicit Printer(NeoOutputStream oss) : m_oss(oss) {}

  template <typename T>
  NeoOutputStream& operator<<(T const& printable) {
    m_oss << printable;
    return m_oss;
  }
};

inline Trace traceLevel() {
  if (std::getenv("NEO_DEBUG_PRINT"))
    return Trace::Verbose;
  else if (std::getenv("NEO_DEBUG_PRINT_IN_FILE"))
    return Trace::VerboseInFile;
  else
    return Trace::Silent;
}

/*!
 *
 * @param rank: current process/subdomain number (0 by default)
 * @return output stream
 *
 * Neo debug logs are activated by environment variables:
 * - NEO_DEBUG_PRINT=1 to activate console debug outputs
 * - NEO_DEBUG_PRINT_IN_FILE=1 to activate file debug outputs (file named Neo_output_rank.txt)
 *
 * Usage: two ways
 * - first:
 * \code
 * auto printer = Neo::printer(my_rank);
 * printer() << printed_data << Neo::endl;
 * \endcode
 * - second
 * \code
 * Neo::printer(my_rank) << printed_data << Neo::endl;
 * \endcode
 */
inline Printer printer(int rank = 0) {
  auto trace_level = Neo::traceLevel();
  switch (trace_level) {
    case Trace::Verbose: return Printer{NeoOutputStream{Neo::Trace::Verbose}};
    case Trace::Silent: return Printer{NeoOutputStream{Neo::Trace::Silent}};
    case Trace::VerboseInFile: return Printer{NeoOutputStream{Neo::Trace::VerboseInFile, rank}};
  }
  return Printer{NeoOutputStream{Neo::Trace::Silent}};
}

inline NeoOutputStream& operator<<(NeoOutputStream& oss, NeoOutputStreamHandler handler) {
  return handler(oss);
}

template <typename T>
NeoOutputStream& operator<<(Neo::NeoOutputStream& oss, T const& printable) {
  oss.stream() << printable;
  return oss;
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
  NeoOutputStream&  _printContainer(Container const& container, NeoOutputStream& oss){
    std::copy(container.begin(), container.end(), std::ostream_iterator<typename Container::value_type>(oss.stream(), " "));
    oss << Neo::endline;
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

namespace Neo
{
  template <typename Container>
  requires std::ranges::range<Container> && (!std::same_as<std::ranges::range_value_t<Container>,char>)
  NeoOutputStream& operator<<(NeoOutputStream& oss, Container const& container) {
    return utils::_printContainer(container, oss);
  }

  template <typename T>
  NeoOutputStream& operator<<(NeoOutputStream& oss, utils::Span<T> const& container) {
    return utils::_printContainer(container, oss);
  }

  template <typename T>
  NeoOutputStream& operator<<(NeoOutputStream& oss, utils::ConstSpan<T> const& container) {
    return utils::_printContainer(container, oss);
  }




  namespace utils
  {
    template <typename Container>
    void printContainer(Container&& container, std::string const& name = "Container", int rank = 0) {
      Neo::NeoOutputStream oss{traceLevel(),rank};
      oss << name << ", size : " << container.size() << Neo::endline;
      _printContainer(container, oss);
    }

    template <typename Container>
    void printContainer(NeoOutputStream& oss, Container&& container, std::string const& name = "Container") {
      oss << name << ", size : " << container.size() << Neo::endline;
      _printContainer(container, oss);
    }
  }
}

//----------------------------------------------------------------------------/
//----------------------------------------------------------------------------/

#endif // NEO_UTILS_H
