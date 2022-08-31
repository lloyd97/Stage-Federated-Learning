#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
from math import ceil
from decimal import Decimal
 
FIELD_SIZE = 10**5


# In[6]:


class Shamir:
    def __init__(self, secret):
        self.secret = secret
            
    def encode_bytes(self,value):
        str_bytes = bytes(value, 'utf-8')
        return str_bytes

    def bytes_to_int(self,value):
        int_val = int.from_bytes(value, "big")
        return int_val

    def int_to_bytes(self,value):
        byte_val = value.to_bytes((value.bit_length() + 7) // 8, 'big')
        return byte_val

    def decode_bytes(self,value):
        decoded_str = value.decode()
        return decoded_str
            
    def generate_secret(self,value):
        return value[:10]
       
    def reconstruct_secret(self,shares):
        """
        Combines individual shares (points on graph)
        using Lagranges interpolation.

        shares is a list of points (x, y) belonging to a
        polynomial with a constant of our key.
        """
        sums = 0
        prod_arr = []

        for j, share_j in enumerate(shares):
            xj, yj = share_j
            prod = Decimal(1)

            for i, share_i in enumerate(shares):
                xi, _ = share_i
                if i != j:
                    prod *= Decimal(Decimal(xi)/(xi-xj))

            prod *= yj
            sums += Decimal(prod)



        return int(round(Decimal(sums), 0))
    
    def polynomial(self,x, coefficients):
        """
        This generates a single point on the graph of given polynomial
        in `x`. The polynomial is given by the list of coefficients.
        """
        poly = 0
        # Loop through reversed list, so that indices from enumerate match the
        # actual coefficient indices
        for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
            poly += x ** coefficient_index * coefficient_value
        return poly
    
    def get_coeff(self,k):
        """
        Randomly generate a list of coefficients for a polynomial with
        degree of k - 1, whose constant is secret.
        """
        coefficient = [random.randrange(0, FIELD_SIZE) for _ in range(k - 1)]
        coefficient.append(self.secret)
        return coefficient
    
    def generate_shares(self,n, k, secret, noise):
        """
        Split given `secret` into `n` shares with minimum threshold
        of `k` shares to recover this `secret`, using SSS(Shamir's Secret Sharing) 
        algorithm.
        """
        self.secret = secret
        coefficients = self.get_coeff(k)
        shares = []

        for i in range(1, n+1):
            x = random.randrange(1, FIELD_SIZE)
            shares.append((x, self.polynomial(x, coefficients)))

        return shares


# In[ ]:




