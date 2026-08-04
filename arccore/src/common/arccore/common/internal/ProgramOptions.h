// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProgramOptions.h                                            (C) 2000-2026 */
/*                                                                           */
/* Program options parser (replacement for boost::program_options).          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_PROGRAMOPTIONS_H
#define ARCCORE_ALINA_PROGRAMOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

#include <any>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::ProgramOptions
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class typed_value;

class options_description;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type-erased base for option value semantics.
 */
class ARCCORE_COMMON_EXPORT option_value
{
 public:

  virtual ~option_value() = default;
  virtual std::any parse(const std::string& s) const = 0;
  virtual std::string default_string() const = 0;
  virtual bool has_default() const = 0;
  virtual bool is_required() const = 0;
  virtual bool is_bool_switch() const = 0;
  virtual bool is_multitoken() const = 0;
  virtual std::any default_value_any() const = 0;
  virtual void assign_bound(const std::any& value) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Typed option value with default, required, and multitoken support.
 */
template <typename T>
class typed_value : public option_value
{
 public:

  typed_value() = default;

  explicit typed_value(T* bound_variable)
  : m_bound_variable(bound_variable)
  {}

  typed_value* default_value(const T& value)
  {
    m_default_value = value;
    m_has_default = true;
    return this;
  }

  typed_value* required()
  {
    m_is_required = true;
    return this;
  }

  typed_value* multitoken()
  {
    m_is_multitoken = true;
    return this;
  }

  std::any parse(const std::string& s) const override
  {
    return std::any(parse_impl(s));
  }

  T parse_impl(const std::string& s) const
  {
    T value{};
    if constexpr (std::is_same_v<T, std::string>) {
      value = s;
    }
    else {
      std::istringstream iss(s);
      iss >> value;
      if (iss.fail())
        throw std::logic_error("Failed to parse option value \"" + s + "\"");
    }
    return value;
  }

  std::string default_string() const override
  {
    if (!m_has_default)
      return {};
    return default_string_impl(m_default_value);
  }

  static std::string default_string_impl(const T& value)
  {
    if constexpr (std::is_same_v<T, std::string>) {
      return value;
    }
    else if constexpr (std::is_same_v<T, bool>) {
      return value ? "true" : "false";
    }
    else {
      std::ostringstream oss;
      oss << value;
      return oss.str();
    }
  }

  bool has_default() const override { return m_has_default; }
  bool is_required() const override { return m_is_required; }
  bool is_bool_switch() const override { return false; }
  bool is_multitoken() const override { return m_is_multitoken; }

  std::any default_value_any() const override
  {
    return std::any(m_default_value);
  }

  void assign_bound(const std::any& value) const override
  {
    if (m_bound_variable)
      *m_bound_variable = std::any_cast<T>(value);
  }

 private:

  T* m_bound_variable = nullptr;
  T m_default_value{};
  bool m_has_default{ false };
  bool m_is_required{ false };
  bool m_is_multitoken{ false };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization for bool switch options.
 */
template <>
class typed_value<bool> : public option_value
{
 public:

  typed_value() = default;

  explicit typed_value(bool* bound_variable)
  : m_bound_variable(bound_variable)
  , m_default_value(false)
  , m_has_default(true)
  , m_is_bool_switch(true)
  {}

  typed_value* default_value(const bool& value)
  {
    m_default_value = value;
    m_has_default = true;
    return this;
  }

  typed_value* required()
  {
    m_is_required = true;
    return this;
  }

  typed_value* multitoken()
  {
    return this;
  }

  std::any parse(const std::string& s) const override
  {
    return std::any(parse_impl(s));
  }

  bool parse_impl(const std::string& s) const
  {
    if (s == "true" || s == "1" || s == "yes" || s == "on")
      return true;
    if (s == "false" || s == "0" || s == "no" || s == "off")
      return false;
    throw std::logic_error("Failed to parse bool option value \"" + s + "\"");
  }

  std::string default_string() const override
  {
    if (!m_has_default)
      return {};
    return m_default_value ? "true" : "false";
  }

  bool has_default() const override { return m_has_default; }
  bool is_required() const override { return m_is_required; }
  bool is_bool_switch() const override { return m_is_bool_switch; }
  bool is_multitoken() const override { return false; }

  std::any default_value_any() const override
  {
    return std::any(m_default_value);
  }

  void assign_bound(const std::any& value) const override
  {
    if (m_bound_variable)
      *m_bound_variable = std::any_cast<bool>(value);
  }

 private:

  bool* m_bound_variable = nullptr;
  bool m_default_value = false;
  bool m_has_default = true;
  bool m_is_bool_switch = true;
  bool m_is_required{ false };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization for std::vector<std::string> (multitoken).
 */
template <>
class typed_value<std::vector<std::string>> : public option_value
{
 public:

  typed_value() = default;

  explicit typed_value(std::vector<std::string>* bound_variable)
  : m_bound_variable(bound_variable)
  {}

  typed_value* default_value(const std::vector<std::string>& value)
  {
    m_default_value = value;
    m_has_default = true;
    return this;
  }

  typed_value* required()
  {
    m_is_required = true;
    return this;
  }

  typed_value* multitoken()
  {
    m_is_multitoken = true;
    return this;
  }

  std::any parse(const std::string& s) const override
  {
    std::vector<std::string> v;
    v.push_back(s);
    return std::any(std::move(v));
  }

  std::string default_string() const override
  {
    if (!m_has_default || m_default_value.empty())
      return {};
    std::string result;
    for (size_t i = 0; i < m_default_value.size(); ++i) {
      if (i > 0)
        result += " ";
      result += m_default_value[i];
    }
    return result;
  }

  bool has_default() const override { return m_has_default; }
  bool is_required() const override { return m_is_required; }
  bool is_bool_switch() const override { return false; }
  bool is_multitoken() const override { return m_is_multitoken; }

  std::any default_value_any() const override
  {
    return std::any(m_default_value);
  }

  void assign_bound(const std::any& value) const override
  {
    if (m_bound_variable)
      *m_bound_variable = std::any_cast<std::vector<std::string>>(value);
  }

 private:

  std::vector<std::string>* m_bound_variable = nullptr;
  std::vector<std::string> m_default_value;
  bool m_has_default{ false };
  bool m_is_required{ false };
  bool m_is_multitoken{ true };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Presence-only option (no value, just a flag).
 *
 * Used for bare flags like ("help,h", "description") where
 * the option has no value semantic.
 */
class ARCCORE_COMMON_EXPORT untyped_value : public option_value
{
 public:

  std::any parse(const std::string&) const override
  {
    return std::any(true);
  }

  std::string default_string() const override { return {}; }

  bool has_default() const override { return false; }
  bool is_required() const override { return false; }
  bool is_bool_switch() const override { return true; }
  bool is_multitoken() const override { return false; }

  std::any default_value_any() const override
  {
    return std::any(false);
  }

  void assign_bound(const std::any&) const override {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Create a typed option value.
 */
template <typename T>
typed_value<T>*
value()
{
  return new typed_value<T>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Create a typed option value bound to a variable.
 */
template <typename T>
typed_value<T>*
value(T* v)
{
  return new typed_value<T>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Create a boolean switch option.
 */
inline typed_value<bool>*
bool_switch()
{
  return new typed_value<bool>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Create a boolean switch option bound to a variable.
 */
inline typed_value<bool>*
bool_switch(bool* v)
{
  return new typed_value<bool>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Internal descriptor for a single option.
 */
struct option_descriptor
{
  std::string long_name;
  std::string short_name;
  std::string description;
  std::shared_ptr<option_value> value_semantic;
  bool has_short{ false };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Describes a set of command-line options.
 */
class ARCCORE_COMMON_EXPORT options_description
{
 public:

  explicit options_description(const std::string& title = "Options");

  /*!
   * \brief Proxy for chained add_options()() calls.
   */
  class ARCCORE_COMMON_EXPORT options_proxy
  {
   public:

    explicit options_proxy(options_description& desc)
    : m_desc(desc)
    {}

    options_proxy& operator()(const char* name, const char* description);
    options_proxy& operator()(const char* name, const typed_value<bool>* value,
                              const char* description);
    options_proxy& operator()(const char* name, const typed_value<bool>* value);

    template <typename T>
    options_proxy& operator()(const char* name, const typed_value<T>* value,
                              const char* description)
    {
      m_desc.add_option(name, std::shared_ptr<option_value>(const_cast<typed_value<T>*>(value)),
                        description);
      return *this;
    }

    template <typename T>
    options_proxy& operator()(const char* name, const typed_value<T>* value)
    {
      m_desc.add_option(name, std::shared_ptr<option_value>(const_cast<typed_value<T>*>(value)),
                        std::string());
      return *this;
    }

   private:

    options_description& m_desc;
  };

  options_proxy add_options();

  void add_option(const std::string& name,
                  std::shared_ptr<option_value> value_semantic,
                  const std::string& description);

  const std::vector<option_descriptor>& options() const { return m_options; }
  const option_descriptor* find(const std::string& name) const;
  const option_descriptor* find_by_short(char short_name) const;

  ARCCORE_COMMON_EXPORT friend std::ostream& operator<<(std::ostream& os, const options_description& desc);

 private:

  std::vector<option_descriptor> m_options;
  std::string m_title;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Stores parsed option values.
 */
class ARCCORE_COMMON_EXPORT variables_map
{
 public:

  class variable_value
  {
   public:

    variable_value() = default;

    // explicit do not build with gcc-12
    variable_value(std::any value, bool is_default = false)
    : m_value(std::move(value))
    , m_is_default(is_default)
    {}

    bool empty() const { return !m_value.has_value(); }
    bool is_default() const { return m_is_default; }

    template <typename T>
    T as() const
    {
      try {
        return std::any_cast<T>(m_value);
      }
      catch (const std::bad_any_cast&) {
        throw std::logic_error("Type mismatch in variable_value::as<T>()");
      }
    }

    void set_semantic(std::shared_ptr<option_value> semantic)
    {
      m_semantic = std::move(semantic);
    }

    const std::shared_ptr<option_value>& semantic() const { return m_semantic; }
    std::any& value_ref() { return m_value; }

   private:

    std::any m_value;
    bool m_is_default{ false };
    std::shared_ptr<option_value> m_semantic;
  };

  void add(const std::string& name, variable_value value);
  bool count(const std::string& name) const;
  const variable_value& operator[](const std::string& name) const;

  void set_semantic(const std::string& name, std::shared_ptr<option_value> semantic);

  using iterator = std::map<std::string, variable_value>::iterator;
  using const_iterator = std::map<std::string, variable_value>::const_iterator;
  iterator begin() { return m_values.begin(); }
  iterator end() { return m_values.end(); }
  const_iterator begin() const { return m_values.begin(); }
  const_iterator end() const { return m_values.end(); }

 private:

  std::map<std::string, variable_value> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Describes positional (non-option) arguments.
 */
class ARCCORE_COMMON_EXPORT positional_options_description
{
 public:

  positional_options_description& add(const std::string& name, int count);

  const std::vector<std::pair<std::string, int>>& options() const { return m_options; }

 private:

  std::vector<std::pair<std::string, int>> m_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Result of parsing command-line arguments.
 */
class ARCCORE_COMMON_EXPORT parsed_options
{
 public:

  struct parsed_option
  {
    std::string name;
    std::vector<std::string> values;
  };

  explicit parsed_options(const options_description* desc)
  : m_desc(desc)
  {}

  const options_description& description() const { return *m_desc; }

  void add_option(const std::string& name, std::vector<std::string> values)
  {
    m_parsed.push_back({ name, std::move(values) });
  }

  const std::vector<parsed_option>& options() const { return m_parsed; }

 private:

  const options_description* m_desc = nullptr;
  std::vector<parsed_option> m_parsed;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fluent command-line parser builder.
 */
class ARCCORE_COMMON_EXPORT command_line_parser
{
 public:

  command_line_parser(int argc, char** argv)
  : m_argc(argc)
  , m_argv(argv)
  {}

  command_line_parser& options(const options_description& desc)
  {
    m_desc = &desc;
    return *this;
  }

  command_line_parser& positional(const positional_options_description& pos)
  {
    m_positional = &pos;
    return *this;
  }

  parsed_options run();

 private:

  int m_argc = 0;
  char** m_argv = nullptr;
  const options_description* m_desc = nullptr;
  const positional_options_description* m_positional = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT parsed_options
parse_command_line(int argc, char** argv, const options_description& desc);

extern "C++" ARCCORE_COMMON_EXPORT void
store(const parsed_options& parsed, variables_map& vm);

extern "C++" ARCCORE_COMMON_EXPORT void
notify(variables_map& vm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::ProgramOptions

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
