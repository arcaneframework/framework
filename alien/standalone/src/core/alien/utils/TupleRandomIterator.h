/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/tuple/tuple.hpp>
#include <iterator>

/*---------------------------------------------------------------------------*/

namespace Alien
{
template <typename Iterator>
struct TupleRandomIteratorTraits
{
  typedef typename Iterator::value_type value_type;
};

template <typename T>
struct TupleRandomIteratorTraits<const T*>
{
  typedef T value_type;
};

template <typename T>
struct TupleRandomIteratorTraits<T*>
{
  typedef T value_type;
};

/*---------------------------------------------------------------------------*/

template <typename IteratorA, typename IteratorB>
class DualRandomIterator
{
 private:
  typedef typename TupleRandomIteratorTraits<IteratorA>::value_type TypeA;
  typedef typename TupleRandomIteratorTraits<IteratorB>::value_type TypeB;

 public: // requiered typedefs (cf std::iterator)
  typedef boost::tuple<TypeA, TypeB> value_type;
  typedef ptrdiff_t difference_type;
  typedef value_type* pointer;
  typedef boost::tuple<TypeA&, TypeB&> reference;
  typedef std::random_access_iterator_tag iterator_category;
  typedef boost::tuple<const TypeA&, const TypeB&> const_reference;

 public:
  DualRandomIterator() {}
  DualRandomIterator(IteratorA i, IteratorB j)
  : m_i(i)
  , m_j(j)
  {}

  DualRandomIterator& operator++()
  {
    ++m_i;
    ++m_j;
    return *this;
  }
  DualRandomIterator& operator--()
  {
    --m_i;
    --m_j;
    return *this;
  }
  DualRandomIterator operator++(int)
  {
    DualRandomIterator tmp(*this);
    operator++();
    return tmp;
  }
  DualRandomIterator operator--(int)
  {
    DualRandomIterator tmp(*this);
    operator--();
    return tmp;
  }

  bool operator==(const DualRandomIterator& rhs) const { return m_i == rhs.m_i; }
  bool operator!=(const DualRandomIterator& rhs) const { return m_i != rhs.m_i; }
  bool operator<(const DualRandomIterator& rhs) const { return m_i < rhs.m_i; }

  reference operator*() { return reference(*m_i, *m_j); }

  difference_type operator-(const DualRandomIterator& rhs) const { return m_i - rhs.m_i; }

  DualRandomIterator& operator+=(const difference_type n)
  {
    m_i += n;
    m_j += n;
    return *this;
  }
  DualRandomIterator& operator-=(const difference_type n)
  {
    m_i -= n;
    m_j -= n;
    return *this;
  }
  DualRandomIterator operator+(const difference_type n) const
  {
    return DualRandomIterator(m_i + n, m_j + n);
  }
  DualRandomIterator operator-(const difference_type n) const
  {
    return DualRandomIterator(m_i - n, m_j - n);
  }

 private:
  IteratorA m_i;
  IteratorB m_j;
};

/*---------------------------------------------------------------------------*/

template <typename IteratorA, typename IteratorB, typename IteratorC>
class TripleRandomIterator
{
 private:
  typedef typename TupleRandomIteratorTraits<IteratorA>::value_type TypeA;
  typedef typename TupleRandomIteratorTraits<IteratorB>::value_type TypeB;
  typedef typename TupleRandomIteratorTraits<IteratorC>::value_type TypeC;

 public: // requiered typedefs (cf std::iterator)
  typedef boost::tuple<TypeA, TypeB, TypeC> value_type;
  typedef ptrdiff_t difference_type;
  typedef value_type* pointer;
  typedef boost::tuple<TypeA&, TypeB&, TypeC&> reference;
  typedef std::random_access_iterator_tag iterator_category;
  typedef boost::tuple<const TypeA&, const TypeB&, const TypeC&> const_reference;

 public:
  TripleRandomIterator() {}
  TripleRandomIterator(IteratorA i, IteratorB j, IteratorC k)
  : m_i(i)
  , m_j(j)
  , m_k(k)
  {}

  TripleRandomIterator& operator++()
  {
    ++m_i;
    ++m_j;
    ++m_k;
    return *this;
  }
  TripleRandomIterator& operator--()
  {
    --m_i;
    --m_j;
    --m_k;
    return *this;
  }
  TripleRandomIterator operator++(int)
  {
    TripleRandomIterator tmp(*this);
    operator++();
    return tmp;
  }
  TripleRandomIterator operator--(int)
  {
    TripleRandomIterator tmp(*this);
    operator--();
    return tmp;
  }

  bool operator==(const TripleRandomIterator& rhs) const { return m_i == rhs.m_i; }
  bool operator!=(const TripleRandomIterator& rhs) const { return m_i != rhs.m_i; }
  bool operator<(const TripleRandomIterator& rhs) const { return m_i < rhs.m_i; }

  reference operator*() { return reference(*m_i, *m_j, *m_k); }

  difference_type operator-(const TripleRandomIterator& rhs) const
  {
    return m_i - rhs.m_i;
  }

  TripleRandomIterator& operator+=(const difference_type n)
  {
    m_i += n;
    m_j += n;
    m_k += n;
    return *this;
  }
  TripleRandomIterator& operator-=(const difference_type n)
  {
    m_i -= n;
    m_j -= n;
    m_k -= n;
    return *this;
  }
  TripleRandomIterator operator+(const difference_type n) const
  {
    return TripleRandomIterator(m_i + n, m_j + n, m_k + n);
  }
  TripleRandomIterator operator-(const difference_type n) const
  {
    return TripleRandomIterator(m_i - n, m_j - n, m_k - n);
  }

 private:
  IteratorA m_i;
  IteratorB m_j;
  IteratorC m_k;
};

/*---------------------------------------------------------------------------*/

template <typename TupleIteratorT>
struct FirstIndexComparator
{
  bool operator()(const typename TupleIteratorT::const_reference& a,
                  const typename TupleIteratorT::const_reference& b) const
  {
    return boost::get<0>(a) < boost::get<0>(b);
  }
};

/*---------------------------------------------------------------------------*/

#if 0 // EMBEDDED TEST

#include <boost/tuple/tuple_io.hpp> // pour l'affichage
#include <iostream>
#include <list>
#include <vector>

int main() {
  int n = 9;

  std::vector<int> a(n);
  typedef std::vector<int>::iterator IteratorA;
  for(int i=0;i<n;++i) a[i] = std::abs(n/2-2*i);
  std::vector<double> b(n);
  typedef std::vector<double>::iterator IteratorB;
  for(int i=0;i<n;++i) b[i] = std::abs(n/2-2*i) + 0.5; 
  std::vector<double> c(n);
  typedef std::vector<double>::iterator IteratorC;  
  for(IteratorC i=c.begin(); i!=c.end();++i) *i = std::abs(n/2-2*std::distance(c.begin(),i)) + 0.5; 

//   typedef DualRandomIterator<IteratorA, IteratorB> TupleIterator;
//   TupleIterator begin(a.begin(), b.begin());
//   TupleIterator end(a.end(), b.end());
//   FirstIndexComparator<TupleIterator> comparator;

  typedef TripleRandomIterator<IteratorC, IteratorA, IteratorB> TupleIterator;
  TupleIterator begin(c.begin(), a.begin(), b.begin());
  TupleIterator end(c.end(), a.end(), b.end());
  FirstIndexComparator<TupleIterator> comparator;

  std::sort(begin, end, comparator);
  
  for(TupleIterator i(begin), ie(end);i!=ie;++i)
    std::cout << *i << std::endl;

  for(int i=0;i<n;++i)
    std::cout << "#" << i << " : " << a[i] << " " << b[i] << std::endl;

  TupleIterator finder1 = std::lower_bound(begin, end,
                                           TupleIterator::value_type(1,1,1),
                                           comparator);
  std::cout << "Finder1 is " << (*finder1) << "\n";

  TupleIterator finder2 = std::lower_bound(begin, end,
                                           TupleIterator::value_type(3,1,1),
                                           comparator);
  std::cout << "Finder2 is " << (*finder2) << "\n";
}
#endif /* EMBEDDED TEST */

} // namespace Alien
