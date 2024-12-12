package com.example.distribute_ui.data
import com.example.distribute_ui.ui.Messaging

val initialMessages = mutableListOf(
    Messaging(
        "Robot",
        "Test",
        "03:07 pm",
        null
    ),
    Messaging(
        "Me",
        "Test Reply",
        "03:07 pm",
        null
    )
)

val exampleModelName = listOf(
    "bloom560m",
    "bloom560m-int8",
    "bloom1b1",
    "bloom1b1-int8",
    "bloom1b7",
    "bloom1b7-int8",
    "bloom3b",
    "bloom3b-int8",
    "bloom7b",
    "bloom7b-int8",
)

val modelMap: HashMap<String, String> = hashMapOf(
    "Bloom" to "bloom560m"
)
