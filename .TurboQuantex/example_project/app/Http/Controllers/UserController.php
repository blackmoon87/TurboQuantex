<?php

namespace App\Http\Controllers;

class UserController
{
    private $db;

    public function __construct($database)
    {
        $this->db = $database;
    }

    public function registerUser($email, $password)
    {
        // Hash password before saving to database
        $hashedPassword = password_hash($password, PASSWORD_BCRYPT);
        
        $sql = "INSERT INTO users (email, password) VALUES (:email, :password)";
        $stmt = $this->db->prepare($sql);
        return $stmt->execute([
            ':email' => $email,
            ':password' => $hashedPassword
        ]);
    }
}
