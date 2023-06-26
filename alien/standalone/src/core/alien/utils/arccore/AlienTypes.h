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

#ifndef ALIEN_UTILS_ARCANE_ALIENTYPES_H
#define ALIEN_UTILS_ARCANE_ALIENTYPES_H

namespace Alien
{

using Arccore::Byte;
using Arccore::Int32;
using Arccore::Int64;
using Arccore::Integer;
using Arccore::Real;
using Arccore::String;

using Arccore::Ref;
using Arccore::SharedArray;
using Arccore::UniqueArray;

using Arccore::ArrayView;
using Arccore::ConstArrayView;

using Arccore::SharedArray2;
using Arccore::UniqueArray2;

using Arccore::Array2View;
using Arccore::ConstArray2View;

using Arccore::MessagePassing::IMessagePassingMng;
using Arccore::MessagePassing::MessageRank;

using Arccore::ISerializer;
using Arccore::MessagePassing::ISerializeMessage;
using Arccore::MessagePassing::ISerializeMessageList;

using Arccore::ITraceMng;
using Arccore::TraceMessage;

using Arccore::ArgumentException;
using Arccore::FatalErrorException;
using Arccore::NotImplementedException;

template <typename T>
void add(UniqueArray<T>& array, const T& value)
{
  array.add(value);
}

template <typename T>
const T*
unguardedBasePointer(const UniqueArray<T>& array)
{
  return array.unguardedBasePointer();
}

template <typename T>
T* unguardedBasePointer(UniqueArray<T>& array)
{
  return array.unguardedBasePointer();
}

template <typename T>
void addRange(UniqueArray<T>& array, ConstArrayView<T> range)
{
  array.addRange(range);
}

template <typename T>
void addRange(UniqueArray<T>& array, T value, Integer size)
{
  array.addRange(value, size);
}

inline char const*
localstr(String const& str)
{
  return str.localstr();
}

template <typename T>
T* dataPtr(UniqueArray<T>& v)
{
  return v.unguardedBasePointer();
}

template <typename T>
T const*
dataPtr(UniqueArray<T> const& v)
{
  return v.unguardedBasePointer();
}

template <typename T>
T* dataPtr(ArrayView<T> v)
{
  return v.begin();
}

template <typename T>
T const*
dataPtr(ConstArrayView<T> v)
{
  return v.begin();
}

template <typename T>
void fill(UniqueArray<T>& v, T value)
{
  v.fill(value);
}

template <typename T>
void fill(ArrayView<T> v, T value)
{
  v.fill(value);
}

template <typename T>
void copy(UniqueArray<T>& v, const ConstArrayView<T>& v2)
{
  v.copy(v2);
}

template <typename T>
void copy(UniqueArray<T>& v, const UniqueArray<T>& v2)
{
  v.copy(v2);
}

template <typename T>
void allocateData(UniqueArray2<T>& v, Integer dim1, Integer dim2)
{
  v.resize(dim1, dim2);
}

template <typename T>
void freeData(UniqueArray<T>& v)
{
  v.dispose();
}

template <typename T>
void fill(UniqueArray2<T>& v, T value)
{
  v.fill(value);
}

template <typename T>
ArrayView<T>
subView(UniqueArray<T>& array, Integer begin, Integer size)
{
  return array.subView(begin, size);
}

template <typename T>
ConstArrayView<T>
subConstView(UniqueArray<T> const& array, Integer begin, Integer size)
{
  return array.subConstView(begin, size);
}

template <typename T>
Array2View<T>
view(UniqueArray2<T>& v)
{
  return v;
}

template <typename T>
void pushBack(UniqueArray<T>& v, T value)
{
  v.add(value);
}

inline bool
isNull(const String& value)
{
  return value.null();
}

template <typename... U>
String
format(const String& format, const U&... value)
{
  return String::format(format, value...);
}
} // namespace Alien

#endif /* ALIEN_UTILS_ARCANE_ALIENTYPES_H */
