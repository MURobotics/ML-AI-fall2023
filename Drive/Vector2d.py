"""
MIT License

Copyright (c) 2021 Oxi-dev0

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import random

class Vector2D(object):
	def __init__(self, _x, _y):
		self.x = _x
		self.y = _y
	
	@staticmethod
	def UnitRandom():
		return Vector2D(random.random(), random.random())
	
	@staticmethod
	def Zero():
		return Vector2D(0,0)
	
	@staticmethod
	def One():
		return Vector2D(1,1)

	def dot(self, other):
		return self.x * other.x + self.y * other.y
	
	def cross(self, other):
		return self.x * other.y - self.y * other.x
	def normal(self):
		return Vector2D(self.y, -self.x)
	
	def __add__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = self.x + other.x
			newY = self.y + other.y
		elif isinstance(other, (int, float)):
			newX = self.x + other
			newY = self.y + other
		else:
			
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __radd__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x + self.x
			newY = other.y + self.y
		elif isinstance(other, (int, float)):
			newX = other + self.x
			newY = other + self.y
		else:
			
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __iadd__(self, other):
		if isinstance(other, Vector2D):
			self.x += other.x
			self.y += other.y
		elif isinstance(other, (int, float)):
			self.x += other
			self.y += other
		else:
			return NotImplemented
		return self
	
	def __sub__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = self.x - other.x
			newY = self.y - other.y
		elif isinstance(other, (int, float)):
			newX = self.x - other
			newY = self.y - other
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __rsub__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x - self.x
			newY = other.y - self.y
		elif isinstance(other, (int, float)):
			newX = other - self.x
			newY = other - self.y
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __isub__(self, other):
		if isinstance(other, Vector2D):
			self.x -= other.x
			self.y -= other.y
		elif isinstance(other, (int, float)):
			self.x -= other
			self.y -= other
		else:
			return NotImplemented
		return self
	
	def __mul__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = self.x * other.x
			newY = self.y * other.y
		elif isinstance(other, (int, float)):
			newX = self.x * other
			newY = self.y * other
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __rmul__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x * self.x
			newY = other.y * self.y
		elif isinstance(other, (int, float)):
			newX = other * self.x
			newY = other * self.y
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __imul__(self, other):
		if isinstance(other, Vector2D):
			self.x *= other.x
			self.y *= other.y
		elif isinstance(other, (int, float)):
			self.x *= other
			self.y *= other
		else:
			return NotImplemented
		return self
	
	def __truediv__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = self.x / other.x
			newY = self.y / other.y
		elif isinstance(other, (int, float)):
			newX = self.x / other
			newY = self.y / other
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __rtruediv__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x / self.x
			newY = other.y / self.y
		elif isinstance(other, (int, float)):
			newX = other / self.x
			newY = other / self.y
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __floordiv__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = self.x // other.x
			newY = self.y // other.y
		elif isinstance(other, (int, float)):
			newX = self.x // other
			newY = self.y // other
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __rfloordiv__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x // self.x
			newY = other.y // self.y
		elif isinstance(other, (int, float)):
			newX = other // self.x
			newY = other // self.y
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __itruediv__(self, other):
		if isinstance(other, Vector2D):
			self.x /= other.x
			self.y /= other.y
		elif isinstance(other, (int, float)):
			self.x /= other
			self.y /= other
		else:
			return NotImplemented
		return self
	
	def __ifloordiv__(self, other):
		if isinstance(other, Vector2D):
			self.x //= other.x
			self.y //= other.y
		elif isinstance(other, (int, float)):
			self.x //= other
			self.y //= other
		else:
			return NotImplemented
		return self
	
	def __pow__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, (int, float)):
			newX = self.x ** other
			newY = self.y ** other
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __rpow__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x ** self.x
			newY = other.y ** self.y
		elif isinstance(other, (int, float)):
			newX = other ** self.x
			newY = other ** self.y
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __neg__(self):
		return Vector2D(-self.x, -self.y)
	
	def __mod__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = self.x % other.x
			newY = self.y % other.y
		elif isinstance(other, (int, float)):
			newX = self.x % other
			newY = self.y % other
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __rmod__(self, other):
		newX = 0
		newY = 0
		if isinstance(other, Vector2D):
			newX = other.x % self.x
			newY = other.y % self.y
		elif isinstance(other, (int, float)):
			newX = other % self.x
			newY = other % self.y
		else:
			return NotImplemented
		return Vector2D(newX, newY)
	
	def __eq__(self, other):
		if isinstance(other, Vector2D):
			return self.x == other.x and self.y == other.y
		else:
			return NotImplemented
	
	def __ne__(self, other):
		if isinstance(other, Vector2D):
			return self.x != other.x or self.y != other.y
		else:
			return NotImplemented
	
	def __gt__(self, other):
		if isinstance(other, Vector2D):
			return self.length > other.length
		else:
			return NotImplemented

	def __ge__(self, other):
		if isinstance(other, Vector2D):
			return self.length >= other.length
		else:
			return NotImplemented

	def __lt__(self, other):
		if isinstance(other, Vector2D):
			return self.length < other.length
		else:
			return NotImplemented

	def __le__(self, other):
		if isinstance(other, Vector2D):
			return self.length <= other.length
		else:
			return NotImplemented
	
	def __neg__(self):
		return Vector2D(-self.x, -self.y)
	
	def __str__(self):
		return "Vector2D {X:" + str(self.x) + ", Y:" + str(self.y) + "}"

	@property
	def length(self):
		return math.sqrt((self.x ** 2) + (self.y ** 2))

	__len__ = length.fget
	
	def getNormalised(self):
		length = self.length

		# Prevent DBZ error
		if length == 0:
			return Vector2D.Zero()

		return self / length
	
	@staticmethod
	def Distance(a, b):
		# (b-a).length
		return math.sqrt(((b.x-a.x)**2)+((b.y-a.y)**2))
	
	@staticmethod
	def Lerp(a, b, t):
		return a + ((b-a).getNormalised() * (Vector2D.Distance(a, b) * t))

	@staticmethod
	def InverseLerp(a, b, v):
		max = (b-a).length
		vDist = (v-a).length
		return vDist / max
	
	@staticmethod
	def DotProduct(a, b):
		if (not isinstance(a, Vector2D)) or (not isinstance(b, Vector2D)):
			raise RuntimeError(f"FVector2D.DotProduct() requires Vector2Ds as parameters.")
		else:
			return a.x * b.x + a.y * b.y
	
	@staticmethod
	def Project(a, b):
		if (not isinstance(a, Vector2D)) or (not isinstance(b, Vector2D)):
			raise RuntimeError(f"Vector2D.Project() requires Vector2Ds as parameters.")
		else:
			normB = b.getNormalised()
			return normB * Vector2D.DotProduct(a, normB)

	def AsInt(self):
		return Vector2D(int(self.x), int(self.y))
	
	@staticmethod
	def isPointOnSegment(p1, p2, p):
		if (not isinstance(p1, Vector2D)) or (not isinstance(p2, Vector2D)) or (not isinstance(p, Vector2D)):
			raise RuntimeError(f"Vector2D.isPointOnSegment() requires Vector2Ds as parameters.")
		else:
			return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)
	
	@staticmethod
	def CrossProduct(a, b):
		if (not isinstance(a, Vector2D)) or (not isinstance(b, Vector2D)):
			raise RuntimeError(f"Vector2D.CrossProduct() requires Vector2Ds as parameters.")
		else:
			return a.x*b.y - a.y*b.x
	
	@staticmethod
	def Direction(p1, p2, p3):
		if (not isinstance(p1, Vector2D)) or (not isinstance(p2, Vector2D)) or (not isinstance(p3, Vector2D)):
			raise RuntimeError(f"Vector2D.Direction() requires Vector2Ds as parameters.")
		else:
			return Vector2D.CrossProduct(p3 - p1, p2 - p1)
	
	@staticmethod
	def isIntersecting(p1, p2, p3, p4):
		if (not isinstance(p1, Vector2D)) or (not isinstance(p2, Vector2D)) or (not isinstance(p3, Vector2D)) or (not isinstance(p4, Vector2D)):
			raise RuntimeError(f"Vector2D.isIntersecting() requires Vector2Ds as parameters.")
		else: 
			d1 = Vector2D.Direction(p3, p4, p1)
			d2 = Vector2D.Direction(p3, p4, p2)
			d3 = Vector2D.Direction(p1, p2, p3)
			d4 = Vector2D.Direction(p1, p2, p4)

			if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
				return True

			elif d1 == 0 and Vector2D.isPointOnSegment(p3, p4, p1):
				return True
			elif d2 == 0 and Vector2D.isPointOnSegment(p3, p4, p2):
				return True
			elif d3 == 0 and Vector2D.isPointOnSegment(p1, p2, p3):
				return True
			elif d4 == 0 and Vector2D.isPointOnSegment(p1, p2, p4):
				return True
			else:
				return False
	
	@staticmethod
	def Intersection(p1, p2, p3, p4):
		if (not isinstance(p1, Vector2D)) or (not isinstance(p2, Vector2D)) or (not isinstance(p3, Vector2D)) or (not isinstance(p4, Vector2D)):
			raise RuntimeError(f"Vector2D.Intersection() requires Vector2Ds as parameters.")
		else:
			if Vector2D.isIntersecting(p1, p2, p3, p4):

				 # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
				 # Calculate intersection point on segments.

				Tn = ((p1.x - p3.x) * (p3.y - p4.y)) - ((p1.y - p3.y) * (p3.x - p4.x))
				Td = ((p1.x - p2.x) * (p3.y - p4.y)) - ((p1.y - p2.y) * (p3.x - p4.x))
				if not Td == 0:
					t = Tn / Td

					x = p1.x + (t * (p2.x - p1.x))
					y = p1.y + (t * (p2.y - p1.y))

					return Vector2D(int(x),int(y))
				else:
					return None
			else:
				return None