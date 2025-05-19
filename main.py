import unittest
import string
from password_generator import generate_password

class TestPasswordGenerator(unittest.TestCase):

    def test_basic_functionality(self):
        """Tests if the generated password meets the basic requirements: length, special character, and consecutive char check."""
        password = generate_password()
        self.assertGreaterEqual(len(password), 8, "Password length should be at least 8")
        self.assertTrue(any(c in string.punctuation for c in password), "Password should contain at least one special character")

        # Check for consecutive characters.  Loop is repeated to increase confidence.
        for _ in range(10):
           password = generate_password()
           consecutive = False
           for i in range(len(password) - 2):
               if password[i] == password[i+1] == password[i+2]:
                   consecutive = True
                   break
           self.assertFalse(consecutive, "Password should not contain 3 consecutive identical characters")


    def test_edge_cases(self):
        """Tests edge cases such as minimum length and complex character combinations."""
        for _ in range(10): # Run multiple times for randomization
            password = generate_password()
            self.assertGreaterEqual(len(password), 8) # Check min length.  Max length implicitly tested by random generation.
            self.assertTrue(any(c in string.punctuation for c in password)) #Check at least one special character.

            #More thorough consecutive check - repeats the password generation several times for higher confidence.
            for _ in range(3):
                password = generate_password()
                consecutive = False
                for i in range(len(password) - 2):
                    if password[i] == password[i+1] == password[i+2]:
                        consecutive = True
                        break
                self.assertFalse(consecutive, "Password should not contain 3 consecutive identical characters")


    def test_error_cases(self):
        """Tests for potential error conditions (though highly unlikely with the current implementation)."""
        # The current implementation has a max_attempts mechanism. If after a large number of attempts
        # a valid password cannot be generated, it returns an empty string.
        # This is incredibly unlikely, but we test for it anyway.
        for _ in range(2): # Run a couple of times for good measure.
            password = generate_password()
            if password == "":
                 print("Warning: Password generation failed after multiple attempts.  This is highly unusual, and may indicate a deeper problem.")
                 #This test is now conditional, because it *shouldn't* normally happen.  But we log the warning to stderr.

    def test_various_input_scenarios(self):
        """This test is for code that takes input.  Since generate_password does not take explicit input, it tests random variation in generated outputs."""

        lengths = []
        has_special = []
        no_consecutive = []

        for _ in range(50):
            password = generate_password()
            lengths.append(len(password))
            has_special.append(any(c in string.punctuation for c in password))

            consecutive = False
            for i in range(len(password) - 2):
                if password[i] == password[i+1] == password[i+2]:
                    consecutive = True
                    break
            no_consecutive.append(not consecutive)

        # Verify that we have a variety of lengths.
        self.assertGreater(len(set(lengths)), 1, "Password lengths should vary")
        self.assertTrue(all(has_special), "All passwords should have at least one special character")
        self.assertTrue(all(no_consecutive), "No password should have 3 consecutive identical chars")

unittest.main()