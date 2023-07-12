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

/*!
 * \file UniverseDataBase.h
 * \brief UniverseDataBase.h
 */

#pragma once

#include <list>
#include <utility>
#include <vector>

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Data base for universe objects
 */
class ALIEN_EXPORT UniverseDataBase final
{
 public:
  //! Key object interface
  class IKey
  {
   protected:
    //! Constructor
    IKey() {}

   public:
    //! Free resources
    virtual ~IKey() {}

   private:
    IKey(const IKey&) = delete;

    IKey(IKey&&) = delete;

    void operator=(const IKey&) = delete;

    void operator=(IKey&&) = delete;
  };

  //! Object interface
  class IObject
  {
   protected:
    //! Constructor
    IObject() {}

   public:
    //! Free resources
    virtual ~IObject() {}

   private:
    IObject(const IObject&) = delete;

    IObject(IObject&&) = delete;

    void operator=(const IObject&) = delete;

    void operator=(IObject&&) = delete;
  };

 private:
  template <typename T>
  using key_type =
  typename std::conditional<std::is_copy_constructible<T>::value, T, T&>::type;

  /*!
   * \brief Key object
   * \tparam T The type of the key
   */
  template <typename T>
  struct Key : public IKey
  {
    //! Free resources
    virtual ~Key() {}

    /*!
     * \brief Constructor
     * \para[in] t The key
     */
    Key(T&& t)
    : value(std::move(t))
    {}

    //! The key
    T value;
  };

  /*!
   * \brief Object
   * \tparam T The type of the key
   */
  template <typename T>
  struct Object : public IObject
  {
    //! Free resources
    virtual ~Object() {}

    /*!
     * \brief Constructor
     * \param[in] v The object
     */
    Object(std::shared_ptr<T> v)
    : value(v)
    {}
    //! The object
    std::shared_ptr<T> value;
  };

 public:
  /*!
   * \brief List of objects
   */
  class ObjectList
  {
   public:
    //! Constructor
    ObjectList() {}

    /*!
     * \brief Find or create an object
     * \tparam U The type of the object
     * \tparam T The type of the objects
     * \param[in] t The objects
     * \returns The object and the initialization flag
     */
    template <typename U, typename... T>
    std::pair<std::shared_ptr<U>, bool> findOrCreate(T&... t)
    {
      using object_type = Object<U>;
      for (auto& o : m_objects) {
        auto _o = std::dynamic_pointer_cast<object_type>(o);
        if (_o) {
          return std::make_pair(_o->value, false);
        }
      }
      return std::make_pair(create<U>(t...), true);
    }

    /*!
     * \brief Creates an object
     * \tparam U The type of the object
     * \tparam T The type of the objects
     * \param[in] t The objects
     * \returns The object created
     */
    template <typename U, typename... T>
    std::shared_ptr<U> create(T&... t)
    {
      auto u = std::make_shared<U>(t...);
      std::shared_ptr<IObject> o(new Object<U>(u));
      m_objects.push_back(o);
      return u;
    }

   private:
    //! The list of objects
    std::list<std::shared_ptr<IObject>> m_objects;
  };

 public:
  //! Constructor
  UniverseDataBase() {}

  /*!
   * \brief Finds or creates an object
   * \tparam U The type of the object
   * \tparam T The type of the objects
   * \param[in] t The objects
   * \returns The object and the initialization flag
   */
  template <typename U, typename... T>
  std::pair<std::shared_ptr<U>, bool> findOrCreate(T&... t)
  {
    using type = std::tuple<key_type<T>...>;
    using key_type = Key<type>;
    type v{ t... };
    for (auto i = 0u; i < m_keys.size(); ++i) {
      auto _k = std::dynamic_pointer_cast<key_type>(m_keys[i]);
      if (_k && std::operator==(_k->value, v)) {
        return m_objects[i]->findOrCreate<U>(t...);
      }
    }
    m_keys.push_back(std::shared_ptr<IKey>(new key_type{ std::move(v) }));
    auto objects = std::make_shared<ObjectList>();
    m_objects.push_back(objects);
    return std::make_pair(objects->create<U>(t...), true);
  }

 private:
  //! The list of keys
  std::vector<std::shared_ptr<IKey>> m_keys;
  //! The list of objects
  std::vector<std::shared_ptr<ObjectList>> m_objects;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
