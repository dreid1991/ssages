/**
 * This file has been obtained from
 * SAPHRON - Statistical Applied PHysics through Random On-the-fly Numerics
 * https://github.com/hsidky/SAPHRON
 *
 * Copyright 2016 Hythem Sidky
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. 
*/
#pragma once 

#include <cmath>
#include "Requirement.h"

namespace Json
{
	//! Requirements on a numeric value
	/*!
	 * The numbers are stored internally as \c double.
	 *
	 * \ingroup Json
	 */
	class NumberRequirement : public Requirement
	{
	private:
		std::string _path; //!< JSON path.
		double _multipleOf; //!< Base value for "multiple of" requirement.
		double _min; //!< Lower bound for range requirement.
		double _max; //!< Upper bound for range requirement.
		bool _multSet; //!< If \c True, "Multiple of" requirement is active.
		bool _minSet; //!< If \c True, Lower bound for range requirement is active.
		bool _maxSet; //!< If \c True, Upper bound for range requirement is active.
		bool _exclMin; //!< If \c True, lower bound is exclusive.
		bool _exclMax; //!< If \c True, upper bound is exclusive.


	public:
		//! Constructor.
		NumberRequirement() : 
		_path(), _multipleOf(0), _min(0), _max(0), _multSet(false), 
		_minSet(false), _maxSet(false), _exclMin(false), _exclMax(false)
		{}

		//! Reset Requirement.
		virtual void Reset() override
		{
			_multipleOf = 0;
			_minSet = _maxSet = false;
			_exclMin = _exclMax = false; 
			_min = _max = 0;
			_multSet = false;
			ClearErrors();
			ClearNotices();
		}

		//! Parse JSON value to set up Requirement.
		/*!
		 * \param json JSON input value.
		 * \param path Path for JSON path specification.
		 */
		virtual void Parse(Value json, const std::string& path) override
		{
			Reset();
			
			_path = path;
			if(json.isMember("multipleOf") && json["multipleOf"].isNumeric())
			{
				_multSet = true;
				_multipleOf = json["multipleOf"].asDouble();
			}

			if(json.isMember("minimum") && json["minimum"].isNumeric())
			{
				_minSet = true;
				_min = json["minimum"].asDouble();
			}

			if(json.isMember("maximum") && json["maximum"].isNumeric())
			{
				_maxSet = true;
				_max = json["maximum"].asDouble();
			}

			if(json.isMember("exclusiveMinimum") && json["exclusiveMinimum"].isBool())
			{
				_exclMin = json["exclusiveMinimum"].asBool();
			}

			if(json.isMember("exclusiveMaximum") && json["exclusiveMaximum"].isBool())
			{
				_exclMax = json["exclusiveMaximum"].asBool();
			}
		}

		//! Validate JSON value.
		/*!
		 * \param json JSON value to validate.
		 * \param path Path for JSON path specification.
		 *
		 * Test that the JSON value meets the requirements set via
		 * NumberRequirement::Parse(). If the validation fails, an error is added
		 * to the list of error messages.
		 */
		virtual void Validate(const Value& json, const std::string& path) override
		{
			if(!json.isNumeric())
			{
				PushError(path + ": Must be of type \"number\".");
				return;
			}

			if(_multSet && fmod(json.asDouble(), _multipleOf) != 0)
				PushError(path + ": Value must be a multiple of " + std::to_string(_multipleOf));

			if(_minSet)
			{
				if(_exclMin && json.asDouble() <= _min)
					PushError(path + ": Value must be greater than " + std::to_string(_min));
				else if(json.asDouble() < _min)
					PushError(path + ": Value cannot be less than " + std::to_string(_min));
			}

			if(_maxSet)
			{
				if(_exclMax && json.asDouble() >= _max)
					PushError(path + ": Value must be less than " + std::to_string(_max));
				else if(json.asDouble() > _max)
					PushError(path + ": Value cannot be greater than " + std::to_string(_max));
			}
		}
	};
}